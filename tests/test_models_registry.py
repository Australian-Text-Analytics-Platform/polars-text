"""Phase 1.7-1.8 tests: prefetch_model + list_loaded_models + models.py."""

from __future__ import annotations

import polars_text as pt
from polars_text.models import (
    RECOMMENDED_TOKENIZERS,
    list_loaded_models,
    prefetch_model,
    recommended_tokenizer_for,
)


def test_recommended_tokenizers_has_core_languages() -> None:
    for lang in ("en", "zh", "ja", "multi", "fallback"):
        assert lang in RECOMMENDED_TOKENIZERS
        assert "/" in RECOMMENDED_TOKENIZERS[lang] or "-" in RECOMMENDED_TOKENIZERS[lang]


def test_recommended_tokenizer_for_known_languages() -> None:
    assert recommended_tokenizer_for("en") == "bert-base-uncased"
    assert recommended_tokenizer_for("zh") == "bert-base-chinese"


def test_recommended_tokenizer_for_unknown_falls_back_to_multi() -> None:
    assert (
        recommended_tokenizer_for("klingon")
        == RECOMMENDED_TOKENIZERS["multi"]
    )


def test_prefetch_model_populates_registry() -> None:
    target = "bert-base-uncased"
    prefetch_model(target)
    assert target in list_loaded_models()


def test_prefetch_model_is_idempotent() -> None:
    target = "bert-base-uncased"
    prefetch_model(target)
    first = list_loaded_models()
    prefetch_model(target)
    second = list_loaded_models()
    # Second call must not duplicate the entry.
    assert first.count(target) == second.count(target) == 1


def test_helpers_exported_from_package_root() -> None:
    # The package-level names should match what models.py exports.
    assert pt.recommended_tokenizer_for is recommended_tokenizer_for
    assert pt.prefetch_model is prefetch_model
    assert pt.list_loaded_models is list_loaded_models
    assert pt.RECOMMENDED_TOKENIZERS is RECOMMENDED_TOKENIZERS
