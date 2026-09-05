from __future__ import annotations

from typing import Any, cast

import polars as pl
import pytest

import polars_text  # noqa: F401


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        pytest.param(
            ["Hello, world!", None],
            [
                [
                    {"token": "hello", "start": 0, "end": 5},
                    {"token": "world", "start": 7, "end": 12},
                ],
                [],
            ],
            id="mixed",
        ),
        pytest.param([None, "", "  "], [[], [], []], id="empty-and-null"),
        pytest.param([], [], id="zero-rows"),
    ],
)
def test_tokenize_values_and_schema(texts, expected) -> None:
    df = pl.DataFrame({"text": texts}, schema={"text": pl.String})
    expr = cast(Any, pl.col("text")).text.tokenize(model="native:plain_words_en")
    out = df.select(expr)
    assert out.schema == {
        "text": pl.List(
            pl.Struct({"token": pl.String, "start": pl.Int64, "end": pl.Int64})
        )
    }
    assert out["text"].to_list() == expected
    assert df.lazy().select(expr).collect_schema() == out.schema


@pytest.mark.parametrize("model", [None, "", "  "], ids=["null", "empty", "blank"])
def test_tokenize_requires_nonblank_model_id(model) -> None:
    with pytest.raises(ValueError, match="requires an explicit model ID"):
        cast(Any, pl.col("text")).text.tokenize(model=model)
