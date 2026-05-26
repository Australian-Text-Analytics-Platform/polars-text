from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import duckdb
import polars as pl
import polars_text as pt
import polars_text.token_cache as token_cache


def test_cached_tokenize_matches_uncached_output(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    df = pl.DataFrame({"text": ["hello world", "hello again", None]})

    uncached = df.select(pt.tokenize(pl.col("text"))).to_dicts()
    cached = df.select(pt.tokenize(pl.col("text"), cache=cache_path)).to_dicts()

    assert cached == uncached
    assert cache_path.exists()


def test_cache_schema_has_six_columns(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    df = pl.DataFrame({"text": ["hello world"]})
    df.select(pt.tokenize(pl.col("text"), cache=cache_path))

    with duckdb.connect(str(cache_path), read_only=True) as conn:
        rows = conn.execute("DESCRIBE token_cache").fetchall()

    assert [row[0] for row in rows] == [
        "model",
        "params_hash",
        "content_hash",
        "tokens",
        "start_offsets",
        "end_offsets",
    ]


def test_warm_cache_does_not_retokenize(tmp_path: Path, monkeypatch: Any) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame({"text": ["hello world", "hello world"]}).lazy()
    expr = pt.tokenize(pl.col("text"), cache=cache_path)
    base.with_columns(expr.alias("tokens")).collect()

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("warm-cache run should not tokenize misses")

    monkeypatch.setattr(token_cache, "_tokenize_misses", fail)

    warm = cast(pl.DataFrame, base.with_columns(expr.alias("tokens")).collect())
    assert warm.height == 2
    assert warm.to_dicts()[0]["tokens"][0]["token"]


def test_filter_pushdown_only_tokenizes_surviving_rows(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "text": ["alpha", "beta", "gamma", "delta", "epsilon"],
        }
    ).lazy()

    seen: list[list[str]] = []
    real = token_cache._tokenize_misses

    def spy(texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        seen.append(list(texts))
        return real(texts, **kwargs)

    monkeypatch.setattr(token_cache, "_tokenize_misses", spy)

    expr = pt.tokenize(pl.col("text"), cache=cache_path)
    filtered = (
        base.with_columns(expr.alias("tokens")).filter(pl.col("id") == 3).collect()
    )

    assert cast(pl.DataFrame, filtered).height == 1
    flat = sorted(text for batch in seen for text in batch)
    assert flat == ["gamma"]


def test_repeated_texts_in_chunk_are_deduplicated(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame({"text": ["same", "same", "same"]}).lazy()

    seen: list[list[str]] = []
    real = token_cache._tokenize_misses

    def spy(texts: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        seen.append(list(texts))
        return real(texts, **kwargs)

    monkeypatch.setattr(token_cache, "_tokenize_misses", spy)

    expr = pt.tokenize(pl.col("text"), cache=cache_path)
    out = cast(pl.DataFrame, base.with_columns(expr.alias("tokens")).collect())

    assert out.height == 3
    flat = [text for batch in seen for text in batch]
    assert flat == ["same"]
