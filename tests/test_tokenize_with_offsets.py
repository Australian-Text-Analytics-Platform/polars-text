"""Phase 2.2 — `tokenize` emits `{token, start, end}` structs.

Offsets are character positions into the lowercased (if `lowercase=True`,
the default) processed text. The schema is `List[Struct{token: String,
start: Int64, end: Int64}]` and forms the contract Phase 2 uses for the
persisted tokenization column on workspace nodes.
"""

from __future__ import annotations

import os
from typing import Any, cast

import polars as pl
import polars_text
import pytest

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


def test_schema_is_list_of_struct() -> None:
    df = pl.DataFrame({"text": ["Hello"]})
    out = df.select(
        cast(Any, pl.col("text")).text.tokenize(model="native:plain_words_en")
    )
    dtype = out.schema["text"]
    assert isinstance(dtype, pl.List)
    inner = dtype.inner
    assert isinstance(inner, pl.Struct)
    field_names = {f.name for f in inner.fields}
    assert field_names == {"token", "start", "end"}


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
        tok_stripped = tok[2:] if tok.startswith("##") else tok
        assert tok_stripped == extracted, (
            f"HF offset mismatch: token={tok!r}, stripped={tok_stripped!r}, "
            f"extracted={extracted!r}, row={row}"
        )


def test_model_is_required() -> None:
    df = pl.DataFrame({"text": ["hello world"]})
    try:
        df.select(cast(Any, pl.col("text")).text.tokenize())
    except TypeError as exc:
        assert "model" in str(exc)
    else:
        raise AssertionError("tokenize should require a model")


@pytest.mark.network
@_requires_lindera_jieba
def test_offsets_are_monotonically_nondecreasing_for_jieba() -> None:
    # Jieba word tokens shouldn't overlap and should advance through the text.
    rows = _structs("他来到了北京清华大学", model="lindera:jieba")
    prev_end = 0
    for row in rows:
        assert row["start"] >= prev_end, row
        assert row["end"] > row["start"], row
        prev_end = row["end"]


def test_empty_text_returns_empty_list() -> None:
    df = pl.DataFrame({"text": [""]})
    out = df.select(
        cast(Any, pl.col("text")).text.tokenize(model="native:plain_words_en")
    )
    rows = out["text"].to_list()[0]
    # Should be empty or a list of zero structs.
    assert list(rows) == []


def test_null_text_in_mixed_column_returns_empty_list() -> None:
    # All-null columns trip polars dtype inference (column becomes null dtype,
    # not String); the plugin requires String input. Mirror the existing
    # tokenize / concordance test pattern: at least one non-null value so the
    # column is inferred String, then verify the None row produces an empty
    # token list.
    df = pl.DataFrame({"text": ["Hello", None]})
    out = df.select(
        cast(Any, pl.col("text")).text.tokenize(model="native:plain_words_en")
    )
    rows = out["text"].to_list()
    assert len(rows) == 2
    assert list(rows[1]) == []
