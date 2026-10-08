"""Progress reporting through ``progress_path`` (Wordflow issue 350)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import polars as pl
import polars_text  # noqa: F401

MODEL_ID = "native:plain_words_en"


def test_tokenize_reports_every_document(tmp_path: Path) -> None:
    progress = tmp_path / "progress.json"
    df = pl.DataFrame({"text": ["hello world", None, "hello again", ""]})

    df.select(
        cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID, progress_path=progress)
    )

    written = json.loads(progress.read_text())
    assert written["label"] == "tokenizing"
    assert written["unit"] == "documents"
    assert written["done"] == 4
    assert written["total"] is None


def test_cached_tokenize_counts_cached_documents(tmp_path: Path) -> None:
    cache = tmp_path / "tokens.duckdb"
    df = pl.DataFrame({"text": ["hello world", "hello world", "hello again"]})
    expr = cast(Any, pl.col("text")).text.tokenize
    df.select(expr(model=MODEL_ID, cache=cache))

    progress = tmp_path / "second.json"
    df.select(expr(model=MODEL_ID, cache=cache, progress_path=progress))

    assert json.loads(progress.read_text())["done"] == 3


def test_no_progress_path_writes_nothing(tmp_path: Path) -> None:
    pl.DataFrame({"text": ["hello"]}).select(
        cast(Any, pl.col("text")).text.tokenize(model=MODEL_ID)
    )
    assert list(tmp_path.iterdir()) == []
