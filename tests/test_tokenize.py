from __future__ import annotations

from typing import Any, cast

import polars as pl
import polars_text as pt


def test_tokenize_expr() -> None:
    df = pl.DataFrame({"text": ["Hello, world!", None]})
    out = df.select(pt.tokenize(pl.col("text"), model="jieba"))
    assert out.shape == (2, 1)
    rows = out["text"].to_list()
    assert rows[0][0]["token"]
    assert list(rows[1]) == []


def test_tokenize_namespace() -> None:
    df = pl.DataFrame({"text": ["Hello, world!", None]})
    text_expr = cast(Any, pl.col("text"))
    out = df.select(text_expr.text.tokenize(model="jieba"))
    assert out.shape == (2, 1)
    rows = out["text"].to_list()
    assert rows[0][0]["token"]
    assert list(rows[1]) == []
