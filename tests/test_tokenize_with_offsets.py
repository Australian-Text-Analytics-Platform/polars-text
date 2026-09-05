"""`tokenize` emits `{token, start, end}` structs.

Offsets are character positions into the lowercased (if `lowercase=True`,
the default) processed text. The schema is `List[Struct{token: String,
start: Int64, end: Int64}]` and forms the contract used for the persisted
tokenization column on workspace nodes.
"""

from __future__ import annotations

import os
from typing import Any, cast

import polars as pl
import pytest

import polars_text  # noqa: F401

_LINDERA_JIEBA_TESTS_ENV = "POLARS_TEXT_RUN_LINDERA_JIEBA_TESTS"
_requires_lindera_jieba = pytest.mark.skipif(
    _LINDERA_JIEBA_TESTS_ENV not in os.environ,
    reason=(
        f"Set {_LINDERA_JIEBA_TESTS_ENV}=1 and provide a reachable "
        "lindera:jieba dictionary archive to run Jieba download tests."
    ),
)


def _structs(text: str, *, model: str | None) -> list[dict]:
    df = pl.DataFrame({"text": [text]})
    out = df.select(cast(Any, pl.col("text")).text.tokenize(model=model))
    rows = out["text"].to_list()[0]
    return list(rows)


@pytest.mark.network
@_requires_lindera_jieba
def test_jieba_offsets_reconstruct_chinese() -> None:
    text = "我爱中国"
    rows = _structs(text, model="lindera:jieba")
    assert rows, "Jieba returned no tokens"
    # Default lowercase=True doesn't change Chinese chars; reconstruction
    # via char-slice must match the token string.
    for row in rows:
        extracted = text[row["start"] : row["end"]]
        assert row["token"] == extracted, (
            f"Jieba offset mismatch: token={row['token']!r}, "
            f"extracted={extracted!r}, row={row}"
        )


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("POLARS_TEXT_RUN_HF_TESTS") != "1",
    reason="Set POLARS_TEXT_RUN_HF_TESTS=1 to exercise the remote Hugging Face tokenizer",
)
def test_hf_offsets_reconstruct_english_lowercased() -> None:
    text = "Tokenization happens fast"
    rows = _structs(text, model="huggingface:bert-base-uncased")
    assert rows, "default HF tokenizer returned no tokens"
    text_lc = text.lower()
    for row in rows:
        extracted = text_lc[row["start"] : row["end"]]
        # WordPiece subwords carry a "##" prefix in the token string but the
        # offsets index the original (un-prefixed) substring.
        tok = row["token"]
        tok_stripped = tok.removeprefix("##")
        assert tok_stripped == extracted, (
            f"HF offset mismatch: token={tok!r}, stripped={tok_stripped!r}, "
            f"extracted={extracted!r}, row={row}"
        )


@pytest.mark.network
@_requires_lindera_jieba
def test_offsets_are_monotonically_nondecreasing_for_jieba() -> None:
    # Jieba word tokens shouldn't overlap and should advance through the text.
    rows = _structs("他来到了北京清华大学", model="lindera:jieba")
    assert rows
    prev_end = 0
    for row in rows:
        assert row["start"] >= prev_end, row
        assert row["end"] > row["start"], row
        prev_end = row["end"]
