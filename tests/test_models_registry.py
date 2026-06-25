from __future__ import annotations

import os

import polars_text as pt
import pytest
from polars_text.models import (
    LINDERA_MODELS_BY_LANGUAGE,
    PREDEFINED_MODELS,
    list_loaded_models,
    prefetch_model,
)

_HF_TESTS_ENV = "POLARS_TEXT_RUN_HF_TESTS"
_requires_hf_models = pytest.mark.skipif(
    _HF_TESTS_ENV not in os.environ,
    reason=f"Set {_HF_TESTS_ENV}=1 to run Hugging Face model registry integration tests.",
)


def test_predefined_models_lists_supported_languages() -> None:
    assert set(PREDEFINED_MODELS) == {
        "native:plain_words_en",
        "huggingface:bert-base-uncased",
        "lindera:cc-cedict",
        "lindera:ja-ipadic",
        "lindera:ja-ipadic-neologd",
        "lindera:ja-unidic",
        "lindera:jieba",
        "lindera:ko-dic",
    }
    assert PREDEFINED_MODELS["native:plain_words_en"] == ("en",)
    assert PREDEFINED_MODELS["huggingface:bert-base-uncased"] == ("en",)
    assert PREDEFINED_MODELS["lindera:cc-cedict"] == ("zh",)
    assert PREDEFINED_MODELS["lindera:jieba"] == ("zh",)
    assert PREDEFINED_MODELS["lindera:ja-ipadic"] == ("ja",)
    assert PREDEFINED_MODELS["lindera:ja-ipadic-neologd"] == ("ja",)
    assert PREDEFINED_MODELS["lindera:ja-unidic"] == ("ja",)
    assert PREDEFINED_MODELS["lindera:ko-dic"] == ("ko",)


def test_lindera_models_by_language_lists_all_supported_dicts() -> None:
    assert LINDERA_MODELS_BY_LANGUAGE == {
        "zh": ("lindera:cc-cedict", "lindera:jieba"),
        "ja": (
            "lindera:ja-ipadic",
            "lindera:ja-ipadic-neologd",
            "lindera:ja-unidic",
        ),
        "ko": ("lindera:ko-dic",),
    }
    for models in LINDERA_MODELS_BY_LANGUAGE.values():
        for model_id in models:
            assert model_id in PREDEFINED_MODELS
            assert model_id.startswith("lindera:")


@pytest.mark.network
@_requires_hf_models
def test_prefetch_model_populates_registry() -> None:
    target = "huggingface:bert-base-uncased"
    prefetch_model(target)
    assert target in list_loaded_models()


@pytest.mark.network
@_requires_hf_models
def test_prefetch_model_is_idempotent() -> None:
    target = "huggingface:bert-base-uncased"
    prefetch_model(target)
    first = list_loaded_models()
    prefetch_model(target)
    second = list_loaded_models()
    # Second call must not duplicate the entry.
    assert first.count(target) == second.count(target) == 1


def test_helpers_exported_from_package_root() -> None:
    # The package-level names should match what models.py exports.
    assert pt.LINDERA_MODELS_BY_LANGUAGE is LINDERA_MODELS_BY_LANGUAGE
    assert pt.prefetch_model is prefetch_model
    assert pt.list_loaded_models is list_loaded_models
    assert pt.PREDEFINED_MODELS is PREDEFINED_MODELS
