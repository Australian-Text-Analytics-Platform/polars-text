"""Public calls on installed full and reduced-feature wheels, without downloads."""

from typing import Any, cast

import polars as pl
import pytest

import polars_text  # noqa: F401
from polars_text._internal import compiled_features


@pytest.mark.parametrize(
    "feature", ["tokenization", "quotation", "embedding", "topic-modeling"]
)
def test_installed_feature_dispatch(feature):
    calls = {
        "tokenization": lambda: cast(Any, pl.col("text")).text.tokenize(
            model="native:plain_words_en"
        ),
        "quotation": lambda: cast(Any, pl.col("text")).text.quotation(
            model_path="/unused/model"
        ),
        "embedding": lambda: cast(Any, pl.col("text")).text.embedding(),
        "topic-modeling": lambda: cast(Any, pl.col("text")).text.topic_modeling(),
    }
    if feature not in compiled_features():
        with pytest.raises(RuntimeError, match=f"requires the '{feature}' feature"):
            calls[feature]()
    else:
        frame = pl.DataFrame({"text": [""]})
        expr = calls[feature]()
        assert "text" in frame.lazy().select(expr).collect_schema()
        if feature in {"tokenization", "quotation"}:
            assert frame.select(expr)["text"].to_list() == [[]]
