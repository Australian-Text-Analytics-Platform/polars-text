"""Recommended tokenizer models per language + prefetch / inventory helpers.

This module is intentionally small. It captures the curated set of tokenizer
model IDs plus thin Python wrappers around the Rust tokenizer registry so the
backend can pre-warm the cache or report which models are loaded.
"""

from __future__ import annotations

from typing import Final

from ._internal import (
    loaded_tokenizers as _loaded_tokenizers,
)
from ._internal import (
    prefetch_tokenizer as _prefetch_tokenizer,
)

PREDEFINED_MODELS: Final[list[str]] = [
    "plain_words_en",
    "jieba",
    "lindera-ja-ipadic",
    "lindera-ja-unidic",
    "lindera-ko-dic",
]


#: Curated tokenizer model IDs per language.
#:
#: - ``en`` uses ``plain_words_en`` for fast local rule-based word tokens.
#:   Hugging Face IDs such as ``bert-base-uncased`` remain valid explicit
#:   model choices when subword tokenization is desired.
#: - ``zh`` uses Lindera Jieba for word-level Chinese segmentation.
#:   Character-level Chinese (``bert-base-chinese``) is intentionally NOT
#:   the default because per-character tokens have little linguistic
#:   meaning in practice. Users can still opt into ``bert-base-chinese``
#:   explicitly via ``model="bert-base-chinese"``.
#: - ``ja`` uses Lindera + IPADIC (morpheme-level segmentation). The dict
#:   is downloaded on first use to ``~/.cache/ldaca/lindera/``. UniDic is
#:   available as an opt-in alternate via ``model="lindera-ja-unidic"``;
#:   see :data:`RECOMMENDED_JA_DICTS` for the surface shown by the frontend
#:   dict selector. The previous default (``cl-tohoku/bert-base-japanese-v3``)
#:   is removed because it has no ``tokenizer.json`` on HF Hub and depends
#:   on Python ``BertJapaneseTokenizer`` + MeCab.
#: - ``ko`` uses Lindera + ko-dic (morpheme-level Korean). Same on-demand
#:   download path as JA. Replaces the previous ``klue/bert-base`` pick,
#:   which worked but only produced sub-word tokens.
#: - ``multi`` is XLM-R, the recommended multilingual default (broader and
#:   stronger than mBERT for most downstream tasks).
#: - ``fallback`` is mBERT, kept as an explicit second-tier choice for
#:   users who specifically want it.
RECOMMENDED_TOKENIZERS: Final[dict[str, str]] = {
    "en": "plain_words_en",
    "zh": "jieba",
    "ja": "lindera-ja-ipadic",
    "ko": "lindera-ko-dic",
    "multi": "xlm-roberta-base",
    "fallback": "bert-base-multilingual-cased",
}


#: Per-language dict choices surfaced by the frontend selector. Languages
#: with a single canonical dict (zh→jieba, ko→ko-dic) deliberately have no
#: entry here so the selector hides itself for them. JA is the only entry
#: today; UniDic is opt-in because it adds ~100 MB to first-use download.
#:
#: Shape: ``{language: [(model_id, human_label), ...]}``. The first entry
#: in each list is the default (matches the corresponding
#: ``RECOMMENDED_TOKENIZERS`` value).
RECOMMENDED_JA_DICTS: Final[tuple[tuple[str, str], ...]] = (
    ("lindera-ja-ipadic", "IPADIC (recommended, ~15 MB)"),
    ("lindera-ja-unidic", "UniDic (more accurate, ~50 MB)"),
)


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
    "PREDEFINED_MODELS",
    "RECOMMENDED_JA_DICTS",
    "RECOMMENDED_TOKENIZERS",
    "list_loaded_models",
    "prefetch_model",
    "recommended_tokenizer_for",
]
