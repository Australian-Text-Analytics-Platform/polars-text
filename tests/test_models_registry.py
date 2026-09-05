from __future__ import annotations

from typing import Any, cast

import polars as pl

import polars_text as pt


def test_catalogue_records_supply_usable_native_model_ids() -> None:
    ids = [model.model_id for model in pt.TOKENIZER_MODELS]
    assert len(set(ids)) == len(ids)
    assert all(model.label and model.languages for model in pt.TOKENIZER_MODELS)
    native = [
        model for model in pt.TOKENIZER_MODELS if model.model_id.startswith("native:")
    ]
    assert native
    for model in native:
        result = pl.DataFrame({"text": ["Hello world"]}).select(
            cast(Any, pl.col("text")).text.tokenize(model=model.model_id)
        )
        assert [token["token"] for token in result["text"][0].to_list()] == [
            "hello",
            "world",
        ]
