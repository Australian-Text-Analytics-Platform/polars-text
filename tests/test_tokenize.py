from __future__ import annotations

from typing import Any, cast

import polars as pl
import polars_text


def test_tokenize_namespace() -> None:
    df = pl.DataFrame({"text": ["Hello, world!", None]})
    out = df.select(
        cast(Any, pl.col("text")).text.tokenize(model="native:plain_words_en")
    )
    assert out.shape == (2, 1)
    rows = out["text"].to_list()
    assert rows[0][0]["token"]
    assert list(rows[1]) == []
