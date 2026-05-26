"""Phase 2.2 — `tokenize` emits `{token, start, end}` structs.

Offsets are character positions into the lowercased (if `lowercase=True`,
the default) processed text. The schema is `List[Struct{token: String,
start: Int64, end: Int64}]` and forms the contract Phase 2 uses for the
persisted tokenization column on workspace nodes.
"""

from __future__ import annotations

import polars as pl
import polars_text as pt


def _structs(text: str, *, model: str | None) -> list[dict]:
    df = pl.DataFrame({"text": [text]})
    out = df.select(pt.tokenize(pl.col("text"), model=model))
    rows = out["text"].to_list()[0]
    return list(rows)


def test_schema_is_list_of_struct() -> None:
    df = pl.DataFrame({"text": ["Hello"]})
    out = df.select(pt.tokenize(pl.col("text")))
    dtype = out.schema["text"]
    assert isinstance(dtype, pl.List)
    inner = dtype.inner
    assert isinstance(inner, pl.Struct)
    field_names = {f.name for f in inner.fields}
    assert field_names == {"token", "start", "end"}


def test_jieba_offsets_reconstruct_chinese() -> None:
    text = "我爱中国"
    rows = _structs(text, model="jieba")
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
    rows = _structs(text, model=None)  # default = bert-base-uncased
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


def test_explicit_default_matches_implicit_default() -> None:
    a = _structs("hello world", model=None)
    b = _structs("hello world", model="bert-base-uncased")
    assert a == b


def test_offsets_are_monotonically_nondecreasing_for_jieba() -> None:
    # Jieba word tokens shouldn't overlap and should advance through the text.
    rows = _structs("他来到了北京清华大学", model="jieba")
    prev_end = 0
    for row in rows:
        assert row["start"] >= prev_end, row
        assert row["end"] > row["start"], row
        prev_end = row["end"]


def test_empty_text_returns_empty_list() -> None:
    df = pl.DataFrame({"text": [""]})
    out = df.select(pt.tokenize(pl.col("text")))
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
    out = df.select(pt.tokenize(pl.col("text")))
    rows = out["text"].to_list()
    assert len(rows) == 2
    assert list(rows[1]) == []
