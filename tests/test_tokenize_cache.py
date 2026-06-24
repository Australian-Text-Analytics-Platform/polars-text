from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import polars_text
from polars_text._internal import debug_token_cache_snapshot

MODEL_ID = "native:plain_words_en"
type TokenCacheDebugRow = tuple[str, str, str, list[str], list[int], list[int]]


def _cache_columns(cache_path: Path) -> list[str]:
    return debug_token_cache_snapshot(cache_path)[0]


def _cache_rows(cache_path: Path) -> list[TokenCacheDebugRow]:
    return debug_token_cache_snapshot(cache_path)[1]


def test_cached_tokenize_matches_uncached_output(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    df = pl.DataFrame({"text": ["hello world", "hello again", None]})

    uncached = df.select(
        cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID)
    ).to_dicts()
    cached = df.select(
        cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path)
    ).to_dicts()

    assert cached == uncached
    assert cache_path.exists()


def test_cache_schema_has_six_columns(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    df = pl.DataFrame({"text": ["hello world"]})
    df.select(cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path))

    assert _cache_columns(cache_path) == [
        "model",
        "params_hash",
        "content_hash",
        "tokens",
        "start_offsets",
        "end_offsets",
    ]


def test_warm_cache_reuses_existing_rows(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame({"text": ["hello world", "hello world"]}).lazy()
    expr = cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path)
    base.with_columns(expr.alias("tokens")).collect()
    first_rows = _cache_rows(cache_path)

    warm = cast(pl.DataFrame, base.with_columns(expr.alias("tokens")).collect())
    assert warm.height == 2
    assert warm.to_dicts()[0]["tokens"][0]["token"]
    assert _cache_rows(cache_path) == first_rows


def test_filter_pushdown_only_tokenizes_surviving_rows(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "text": ["alpha", "beta", "gamma", "delta", "epsilon"],
        }
    ).lazy()

    expr = cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path)
    filtered = (
        base.with_columns(expr.alias("tokens")).filter(pl.col("id") == 3).collect()
    )

    assert cast(pl.DataFrame, filtered).height == 1
    rows = _cache_rows(cache_path)
    assert len(rows) == 1
    assert rows[0][3] == ["gamma"]


def test_repeated_texts_in_chunk_are_deduplicated(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame({"text": ["same", "same", "same"]}).lazy()

    expr = cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path)
    out = cast(pl.DataFrame, base.with_columns(expr.alias("tokens")).collect())

    assert out.height == 3
    rows = _cache_rows(cache_path)
    assert len(rows) == 1
    assert rows[0][3] == ["same"]


def test_limit_pushdown_only_caches_materialized_rows(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame({"text": ["alpha", "beta", "gamma", "delta"]}).lazy()

    expr = cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path)
    out = cast(pl.DataFrame, base.limit(2).with_columns(expr.alias("tokens")).collect())

    assert out.height == 2
    rows = _cache_rows(cache_path)
    assert sorted(row[3][0] for row in rows) == ["alpha", "beta"]
