from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import polars as pl
import polars_text  # noqa: F401

MODEL_ID = "native:plain_words_en"


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


def test_warm_cache_reuses_existing_rows(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.duckdb"
    base = pl.DataFrame({"text": ["hello world", "hello world"]}).lazy()
    expr = cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, cache=cache_path)
    base.with_columns(expr.alias("tokens")).collect()
    first = cast(pl.DataFrame, base.with_columns(expr.alias("tokens")).collect())

    warm = cast(pl.DataFrame, base.with_columns(expr.alias("tokens")).collect())
    assert warm.height == 2
    assert warm.to_dicts()[0]["tokens"][0]["token"]
    assert warm.to_dicts() == first.to_dicts()
