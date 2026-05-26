"""Phase 1.9: Jieba Chinese backend — word-level segmentation.

Verifies that `model="jieba"` produces word-level Chinese tokens, distinct
from the character-level fallback you get with `bert-base-chinese`.

Jieba's dictionary is bundled with the jieba-rs crate, so these tests do
not require network access.
"""

from __future__ import annotations

from typing import Any, cast

import polars as pl
import polars_text


def _tokens_for(text: str, *, model: str | None) -> list[str]:
    df = pl.DataFrame({"text": [text]})
    out = df.select(cast(Any, pl.col("text")).text.tokenize(model=model))
    return [entry["token"] for entry in out["text"].to_list()[0]]


def test_jieba_produces_word_level_chinese_tokens() -> None:
    # "今天天气很好" = "the weather is nice today"
    # Jieba should segment into 3 words: 今天 / 天气 / 很好 (not 6 chars).
    tokens = _tokens_for("今天天气很好", model="jieba")
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
    jieba_tokens = _tokens_for(text, model="jieba")
    bert_tokens = _tokens_for(text, model="bert-base-chinese")
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
    tokens = _tokens_for("我喜欢 Python 编程", model="jieba")
    assert tokens, f"Jieba returned no tokens for mixed text: got {tokens!r}"
    # The English word should appear intact (possibly lowercased by our pipeline).
    assert any("python" in t.lower() for t in tokens), tokens


def test_jieba_default_for_zh_via_recommended_tokenizer() -> None:
    # End-to-end: looking up the recommended tokenizer for "zh" and using it
    # gives the same result as passing model="jieba" directly.
    recommended = polars_text.recommended_tokenizer_for("zh")
    assert recommended == "jieba"
    a = _tokens_for("今天天气很好", model=recommended)
    b = _tokens_for("今天天气很好", model="jieba")
    assert a == b


def test_jieba_does_not_pollute_english_default() -> None:
    # Loading the Jieba backend must not change the default English tokenizer.
    _tokens_for("我喜欢 Python", model="jieba")
    default_tokens = _tokens_for("Hello, world!", model=None)
    # Same expectation as test_pluggable_tokenizer.py: subwords or alnum-only tokens.
    assert default_tokens
    assert all(t.isalnum() or "##" in t for t in default_tokens), default_tokens
