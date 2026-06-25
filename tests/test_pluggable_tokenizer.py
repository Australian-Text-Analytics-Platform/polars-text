"""Opt-in integration tests for Hugging Face-backed tokenizers.

These tests trigger network fetches from HuggingFace Hub on first run
(same as the existing tokenize / topic-modeling tests). Subsequent runs
use the cached models.

Run locally with ``POLARS_TEXT_RUN_HF_TESTS=1 uv run pytest -m network``.
"""

from __future__ import annotations

import os
from typing import Any, cast

import polars as pl
import polars_text
import pytest

_HF_TESTS_ENV = "POLARS_TEXT_RUN_HF_TESTS"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        _HF_TESTS_ENV not in os.environ,
        reason=f"Set {_HF_TESTS_ENV}=1 to run Hugging Face tokenizer integration tests.",
    ),
]


ALT_MODEL_EN_MULTILINGUAL = "huggingface:bert-base-multilingual-cased"
ALT_MODEL_ZH = "huggingface:bert-base-chinese"


def _tokens_for(text: str, *, model: str | None) -> list[str]:
    df = pl.DataFrame({"text": [text]})
    out = df.select(cast(Any, pl.col("text")).text.tokenize(model=model))
    series = out["text"].to_list()[0]
    return [entry["token"] for entry in series]


def test_bert_uncased_produces_non_empty_tokens() -> None:
    tokens = _tokens_for("Hello, world!", model="huggingface:bert-base-uncased")
    assert tokens, "bert-base-uncased must return non-empty tokens"
    assert all(tok.isalnum() or "##" in tok for tok in tokens), tokens


def test_different_models_produce_different_tokens_for_english() -> None:
    text = "Tokenization differs between BERT models."
    uncased = _tokens_for(text, model="huggingface:bert-base-uncased")
    multilingual = _tokens_for(text, model=ALT_MODEL_EN_MULTILINGUAL)
    assert uncased and multilingual
    assert uncased != multilingual, (
        "huggingface:bert-base-uncased and huggingface:bert-base-multilingual-cased should disagree "
        "on at least one English token (different vocabularies)."
    )


def test_chinese_model_produces_non_empty_tokens_for_chinese() -> None:
    text = "今天天气很好"
    tokens = _tokens_for(text, model=ALT_MODEL_ZH)
    # bert-base-chinese is character-level for Chinese, so each Hanzi becomes
    # one token. We expect at least 4 character-level tokens here (the corpus
    # has 6 chars; punctuation/normalization may drop some).
    assert len(tokens) >= 4, tokens
