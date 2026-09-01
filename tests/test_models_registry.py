from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars_text as pt
import pytest
from polars_text.models import TOKENIZER_MODELS, TokenizerModel


def test_tokenizer_catalog_contains_one_record_per_model() -> None:
    assert [model.model_id for model in TOKENIZER_MODELS] == [
        "native:plain_words_en",
        "huggingface:bert-base-uncased",
        "lindera:jieba",
        "lindera:cc-cedict",
        "lindera:ja-ipadic",
        "lindera:ja-ipadic-neologd",
        "lindera:ja-unidic",
        "lindera:ko-dic",
    ]
    assert len({model.model_id for model in TOKENIZER_MODELS}) == len(TOKENIZER_MODELS)
    assert all(model.label and model.languages for model in TOKENIZER_MODELS)


def test_tokenizer_catalog_is_immutable() -> None:
    model = TOKENIZER_MODELS[0]
    with pytest.raises(FrozenInstanceError):
        model.label = "changed"  # type: ignore[misc]
    assert isinstance(TOKENIZER_MODELS, tuple)


def test_catalog_types_are_exported_from_package_root() -> None:
    assert pt.TOKENIZER_MODELS is TOKENIZER_MODELS
    assert pt.TokenizerModel is TokenizerModel
