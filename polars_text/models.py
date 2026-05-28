"""Predefined tokenizer model inventory + registry helpers.

This module is intentionally small. It captures the curated set of tokenizer
model IDs plus thin Python wrappers around the Rust tokenizer registry so callers
can pre-warm the cache or report which models are loaded.
"""

from __future__ import annotations

from typing import Final

from ._internal import (
    loaded_tokenizers as _loaded_tokenizers,
)
from ._internal import (
    prefetch_tokenizer as _prefetch_tokenizer,
)

#: Predefined model IDs keyed by model ID, with the language codes each model is
#: meant to support. This is inventory only, not a recommendation policy.
PREDEFINED_MODELS: Final[dict[str, tuple[str, ...]]] = {
    "native:plain_words_en": ("en",),
    "huggingface:bert-base-uncased": ("en",),
    "lindera:cc-cedict": ("zh",),
    "lindera:jieba": ("zh",),
    "lindera:ja-ipadic": ("ja",),
    "lindera:ja-ipadic-neologd": ("ja",),
    "lindera:ja-unidic": ("ja",),
    "lindera:ko-dic": ("ko",),
}


#: Human-facing labels for the predefined inventory. Keep labels next to the
#: inventory so API clients do not need to hard-code model catalog metadata.
PREDEFINED_MODEL_LABELS: Final[dict[str, str]] = {
    "native:plain_words_en": "Plain words (English)",
    "huggingface:bert-base-uncased": "BERT base uncased",
    "lindera:cc-cedict": "CC-CEDICT",
    "lindera:jieba": "Jieba",
    "lindera:ja-ipadic": "IPADIC",
    "lindera:ja-ipadic-neologd": "IPADIC Neologd",
    "lindera:ja-unidic": "UniDic",
    "lindera:ko-dic": "ko-dic",
}


#: Lindera dictionary-backed tokenizer IDs grouped by supported language.
LINDERA_MODELS_BY_LANGUAGE: Final[dict[str, tuple[str, ...]]] = {
    "zh": ("lindera:cc-cedict", "lindera:jieba"),
    "ja": (
        "lindera:ja-ipadic",
        "lindera:ja-ipadic-neologd",
        "lindera:ja-unidic",
    ),
    "ko": ("lindera:ko-dic",),
}


def prefetch_model(model_id: str) -> None:
    """Load a tokenizer into the in-process registry.

    Safe to call multiple times — subsequent calls are no-ops. Useful as a
    pre-warm step before an analysis, so the first user-visible tokenize call
    does not block on Hugging Face or Lindera dictionary I/O.
    """
    _prefetch_tokenizer(model_id)


def list_loaded_models() -> list[str]:
    """Return the model IDs currently cached in the tokenizer registry."""
    return list(_loaded_tokenizers())


def predefined_model_records() -> list[dict[str, object]]:
    """Return predefined tokenizer model records for API clients."""
    return [
        {
            "model": model,
            "label": PREDEFINED_MODEL_LABELS.get(model, model),
            "languages": list(languages),
        }
        for model, languages in PREDEFINED_MODELS.items()
    ]


__all__ = [
    "LINDERA_MODELS_BY_LANGUAGE",
    "PREDEFINED_MODELS",
    "PREDEFINED_MODEL_LABELS",
    "list_loaded_models",
    "predefined_model_records",
    "prefetch_model",
]
