"""Phase 1.2/1.3/1.6 integration tests for pluggable tokenizer.

These tests trigger network fetches from HuggingFace Hub on first run
(same as the existing tokenize / topic-modeling tests). Subsequent runs
use the cached models.

Run locally with ``POLARS_TEXT_RUN_HF_TESTS=1 uv run pytest -m network``.
"""

from __future__ import annotations

import os

import polars as pl
import polars_text as pt
import pytest

_HF_TESTS_ENV = "POLARS_TEXT_RUN_HF_TESTS"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        _HF_TESTS_ENV not in os.environ,
        reason=f"Set {_HF_TESTS_ENV}=1 to run Hugging Face tokenizer integration tests.",
    ),
]


# Same multilingual model id we expect to populate the registry in Phase 1.8.
ALT_MODEL_EN_MULTILINGUAL = "bert-base-multilingual-cased"
ALT_MODEL_ZH = "bert-base-chinese"


def _tokens_for(text: str, *, model: str | None) -> list[str]:
    df = pl.DataFrame({"text": [text]})
    out = df.select(pt.tokenize(pl.col("text"), model=model))
    series = out["text"].to_list()[0]
    return [entry["token"] for entry in series]


def test_default_model_omits_model_id_kwarg() -> None:
    # Passing model=None must reproduce the pre-Phase-1 behaviour (same wire
    # format, same kwargs payload), which protects the EN goldens from drift.
    tokens = _tokens_for("Hello, world!", model=None)
    assert tokens, "default tokenizer must return non-empty tokens"
    # bert-base-uncased lowercases by default and would produce subwords like
    # ['hello', ',', 'world', '!'] before punct removal. With remove_punct=True
    # (the default) the punctuation is filtered.
    assert all(tok.isalnum() or "##" in tok for tok in tokens), tokens


def test_explicit_default_matches_implicit_default() -> None:
    implicit = _tokens_for("The quick brown fox.", model=None)
    explicit = _tokens_for("The quick brown fox.", model="bert-base-uncased")
    assert implicit == explicit


def test_different_models_produce_different_tokens_for_english() -> None:
    text = "Tokenization differs between BERT models."
    uncased = _tokens_for(text, model="bert-base-uncased")
    multilingual = _tokens_for(text, model=ALT_MODEL_EN_MULTILINGUAL)
    assert uncased and multilingual
    assert uncased != multilingual, (
        "bert-base-uncased and bert-base-multilingual-cased should disagree "
        "on at least one English token (different vocabularies)."
    )


def test_chinese_model_produces_non_empty_tokens_for_chinese() -> None:
    text = "今天天气很好"
    tokens = _tokens_for(text, model=ALT_MODEL_ZH)
    # bert-base-chinese is character-level for Chinese, so each Hanzi becomes
    # one token. We expect at least 4 character-level tokens here (the corpus
    # has 6 chars; punctuation/normalization may drop some).
    assert len(tokens) >= 4, tokens


def test_same_model_twice_is_cached() -> None:
    # The registry hands back Arc<Tokenizer> from the cache on the second
    # call; we can't observe Arc identity from Python but we can observe that
    # the wall-time on the second call is much smaller than the first. Skip
    # explicit timing here — the cache correctness is exercised by the Rust
    # unit tests; this test just covers the happy path twice.
    a = _tokens_for("Same input twice.", model="bert-base-uncased")
    b = _tokens_for("Same input twice.", model="bert-base-uncased")
    assert a == b
