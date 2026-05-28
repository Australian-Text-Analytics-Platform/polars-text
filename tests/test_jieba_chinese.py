"""Phase 1.9: Lindera Jieba Chinese backend — word-level segmentation.

Verifies that `model="lindera:jieba"` produces word-level Chinese tokens,
distinct from the character-level fallback you get with
`huggingface:bert-base-chinese`.

Jieba is no longer embedded. These tests are gated because they download the
official Lindera Jieba dictionary zip on first use.
"""

from __future__ import annotations

import os
from typing import Any, cast

import polars as pl
import polars_text
import pytest

_LINDERA_JIEBA_MODEL_ID = "lindera:jieba"
_BERT_ZH_MODEL_ID = "huggingface:bert-base-chinese"
_LINDERA_JIEBA_TESTS_ENV = "POLARS_TEXT_RUN_LINDERA_JIEBA_TESTS"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        _LINDERA_JIEBA_TESTS_ENV not in os.environ,
        reason=(
            f"Set {_LINDERA_JIEBA_TESTS_ENV}=1 to run Lindera Jieba download tests."
        ),
    ),
]


def _tokens_for(text: str, *, model: str | None) -> list[str]:
    df = pl.DataFrame({"text": [text]})
    out = df.select(cast(Any, pl.col("text")).text.tokenize(model=model))
    return [entry["token"] for entry in out["text"].to_list()[0]]


def test_jieba_produces_word_level_chinese_tokens() -> None:
    # "今天天气很好" = "the weather is nice today"
    # Jieba should segment into 3 words: 今天 / 天气 / 很好 (not 6 chars).
    tokens = _tokens_for("今天天气很好", model=_LINDERA_JIEBA_MODEL_ID)
    assert tokens, "Jieba returned no tokens"
    # The hallmark of word-level segmentation is that at least one token is
    # multi-character. Char-level fallback would produce only single-char tokens.
    multi_char = [t for t in tokens if len(t) > 1]
    assert multi_char, f"expected word-level (multi-char) tokens, got {tokens!r}"
    # Be permissive about exact segmentation (Jieba's HMM may evolve), but
    # length should be much smaller than the 6-char baseline.
    assert len(tokens) <= 5, f"too many tokens for word-level segmentation: {tokens!r}"


def test_jieba_differs_from_bert_base_chinese() -> None:
    text = (
        "中国人民解放军"  # "Chinese People's Liberation Army" — a single named entity
    )
    jieba_tokens = _tokens_for(text, model=_LINDERA_JIEBA_MODEL_ID)
    bert_tokens = _tokens_for(text, model=_BERT_ZH_MODEL_ID)
    # bert-base-chinese should produce char-level tokens (one token per Hanzi).
    assert all(len(t) == 1 for t in bert_tokens), (
        f"bert-base-chinese should be char-level, got {bert_tokens!r}"
    )
    # Jieba should produce a smaller, word-aware segmentation.
    assert len(jieba_tokens) < len(bert_tokens), (
        f"Jieba should produce fewer tokens than char-level "
        f"(jieba={jieba_tokens!r}, bert={bert_tokens!r})"
    )


def test_jieba_handles_mixed_zh_en_text() -> None:
    tokens = _tokens_for("我喜欢 Python 编程", model=_LINDERA_JIEBA_MODEL_ID)
    assert tokens, f"Jieba returned no tokens for mixed text: got {tokens!r}"
    # The English word should appear intact (possibly lowercased by our pipeline).
    assert any("python" in t.lower() for t in tokens), tokens


def test_jieba_is_exposed_for_zh_inventory() -> None:
    assert _LINDERA_JIEBA_MODEL_ID in polars_text.LINDERA_MODELS_BY_LANGUAGE["zh"]
    assert polars_text.PREDEFINED_MODELS[_LINDERA_JIEBA_MODEL_ID] == ("zh",)


def test_jieba_does_not_pollute_english_default() -> None:
    # Loading the Jieba backend must not change an explicitly loaded English tokenizer.
    _tokens_for("我喜欢 Python", model=_LINDERA_JIEBA_MODEL_ID)
    english_tokens = _tokens_for("Hello, world!", model="native:plain_words_en")
    assert english_tokens
    assert all(t.isalnum() for t in english_tokens), english_tokens
