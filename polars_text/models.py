"""Recommended tokenizer models per language + prefetch / inventory helpers.

This module is intentionally small. It captures the curated set of
HuggingFace tokenizer model IDs we ship Phase 1 with, plus thin Python
wrappers around the Rust registry so the backend can pre-warm the
cache or report which models are loaded.

Phase 1 only exposes the **tokenizer** registry through Python.
POS-tagging and embedder registries (also restructured in Phase 1.4 / 1.5)
get their public surface in Phase 3 (per-tool language routing) and
Phase 3.1 (embedder selection).
"""

from __future__ import annotations

from typing import Final

from ._internal import (
    loaded_tokenizers as _loaded_tokenizers,
    prefetch_tokenizer as _prefetch_tokenizer,
)


#: Curated tokenizer model IDs per language.
#:
#: - ``en`` is the historical default and what the EN goldens were built
#:   against; do not change without regenerating Phase 0.2 goldens.
#: - ``zh`` uses BERT-Chinese, which is character-level (each Hanzi is one
#:   token). For word-level Chinese, switch to Jieba in Phase 5.
#: - ``ja`` uses cl-tohoku's MeCab-aware Japanese BERT — better than mBERT
#:   for Japanese morphology.
#: - ``multi`` is XLM-R, the recommended multilingual default (broader and
#:   stronger than mBERT for most downstream tasks).
#: - ``fallback`` is mBERT, kept as an explicit second-tier choice for
#:   users who specifically want it.
RECOMMENDED_TOKENIZERS: Final[dict[str, str]] = {
    "en": "bert-base-uncased",
    "zh": "bert-base-chinese",
    "ja": "cl-tohoku/bert-base-japanese-v3",
    "multi": "xlm-roberta-base",
    "fallback": "bert-base-multilingual-cased",
}


def recommended_tokenizer_for(language: str) -> str:
    """Return the recommended tokenizer model ID for a language code.

    Falls back to the multilingual default for unknown languages so callers
    never get a KeyError mid-pipeline.
    """
    return RECOMMENDED_TOKENIZERS.get(language, RECOMMENDED_TOKENIZERS["multi"])


def prefetch_model(model_id: str) -> None:
    """Load a tokenizer into the in-process registry (one-time HF Hub fetch).

    Safe to call multiple times — subsequent calls are no-ops. Useful as a
    pre-warm step before an analysis, so the first user-visible tokenize
    call doesn't block on network I/O.
    """
    _prefetch_tokenizer(model_id)


def list_loaded_models() -> list[str]:
    """Return the model IDs currently cached in the tokenizer registry."""
    return list(_loaded_tokenizers())


__all__ = [
    "RECOMMENDED_TOKENIZERS",
    "list_loaded_models",
    "prefetch_model",
    "recommended_tokenizer_for",
]
