"""Phase 1.7-1.8 tests: prefetch_model + list_loaded_models + models.py."""

from __future__ import annotations

import polars_text as pt
from polars_text.models import (
    RECOMMENDED_JA_DICTS,
    RECOMMENDED_TOKENIZERS,
    list_loaded_models,
    prefetch_model,
    recommended_tokenizer_for,
)


def test_recommended_tokenizers_has_core_languages() -> None:
    for lang in ("en", "zh", "ja", "ko", "multi", "fallback"):
        assert lang in RECOMMENDED_TOKENIZERS
        assert RECOMMENDED_TOKENIZERS[lang], (
            f"empty tokenizer id for language {lang!r}"
        )


def test_recommended_tokenizer_for_known_languages() -> None:
    assert recommended_tokenizer_for("en") == "bert-base-uncased"
    assert recommended_tokenizer_for("zh") == "jieba"
    assert recommended_tokenizer_for("ja") == "lindera-ja-ipadic"
    assert recommended_tokenizer_for("ko") == "lindera-ko-dic"


def test_recommended_ja_dicts_starts_with_default_ja_model() -> None:
    # The frontend dict selector relies on the first entry being the
    # current ``RECOMMENDED_TOKENIZERS["ja"]`` so opening the dialog
    # preselects the recommended option.
    assert RECOMMENDED_JA_DICTS[0][0] == RECOMMENDED_TOKENIZERS["ja"]
    # Every entry must be a (model_id, human_label) pair.
    for model_id, label in RECOMMENDED_JA_DICTS:
        assert model_id.startswith("lindera-ja-"), (
            f"unexpected non-Lindera entry in RECOMMENDED_JA_DICTS: {model_id!r}"
        )
        assert label, "human label must be non-empty for the selector"


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
    assert pt.RECOMMENDED_JA_DICTS is RECOMMENDED_JA_DICTS
