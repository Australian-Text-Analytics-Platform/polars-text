import os

import polars as pl
import polars_text as pt
import pytest


def test_quotation_schema() -> None:
    if os.environ.get("POLARS_TEXT_SKIP_POS") == "1":
        pytest.skip("Skipping POS model download for quotation tests")
    df = pl.DataFrame({"text": ['Alice said "Hello world".']})
    out = df.select(pt.quotation(pl.col("text")).alias("quotes"))
    dtype = out.schema["quotes"]
    assert dtype == pl.List(
        pl.Struct([
            pl.Field("speaker", pl.String),
            pl.Field("speaker_start_idx", pl.Int64),
            pl.Field("speaker_end_idx", pl.Int64),
            pl.Field("quote", pl.String),
            pl.Field("quote_start_idx", pl.Int64),
            pl.Field("quote_end_idx", pl.Int64),
            pl.Field("verb", pl.String),
            pl.Field("verb_start_idx", pl.Int64),
            pl.Field("verb_end_idx", pl.Int64),
            pl.Field("quote_type", pl.String),
            pl.Field("quote_token_count", pl.Int64),
            pl.Field("is_floating_quote", pl.Boolean),
        ])
    )


def test_quotation_explode_unnest() -> None:
    if os.environ.get("POLARS_TEXT_SKIP_POS") == "1":
        pytest.skip("Skipping POS model download for quotation tests")
    df = pl.DataFrame({"text": ['Alice said "Hello world".']})
    out = df.select(pl.col("text").text.quotation().list.explode().struct.unnest())
    assert out.height == 1
    assert "quote" in out.columns
    assert out["quote"][0] == '"Hello world"'


def test_quotation_unicode_emoji() -> None:
    """Emoji-heavy text must not panic the plugin."""
    if os.environ.get("POLARS_TEXT_SKIP_POS") == "1":
        pytest.skip("Skipping POS model download for quotation tests")
    df = pl.DataFrame({
        "text": [
            '👩🏼\u200d🌾 said "Great result 😊 for café".',
            "😊👍✊ 🔥 no quotes here 💚",
            "She said \u201cViệt Nam is beautiful\u201d today.",
        ]
    })
    out = df.select(pt.quotation(pl.col("text")).alias("quotes"))
    # Should produce a result without crashing
    assert out.height == 3
    assert out.schema["quotes"] == pl.List(
        pl.Struct([
            pl.Field("speaker", pl.String),
            pl.Field("speaker_start_idx", pl.Int64),
            pl.Field("speaker_end_idx", pl.Int64),
            pl.Field("quote", pl.String),
            pl.Field("quote_start_idx", pl.Int64),
            pl.Field("quote_end_idx", pl.Int64),
            pl.Field("verb", pl.String),
            pl.Field("verb_start_idx", pl.Int64),
            pl.Field("verb_end_idx", pl.Int64),
            pl.Field("quote_type", pl.String),
            pl.Field("quote_token_count", pl.Int64),
            pl.Field("is_floating_quote", pl.Boolean),
        ])
    )


def test_quotation_smart_quotes() -> None:
    """Curly/smart quotes (\u201c \u201d) should be detected correctly."""
    if os.environ.get("POLARS_TEXT_SKIP_POS") == "1":
        pytest.skip("Skipping POS model download for quotation tests")
    df = pl.DataFrame({
        "text": ["He replied \u201cI agree completely\u201d without hesitation."]
    })
    out = df.select(pl.col("text").text.quotation().list.explode().struct.unnest())
    assert out.height >= 1
    assert "quote" in out.columns


def test_quotation_invisible_unicode() -> None:
    """Invisible word joiners / zero-width chars must not crash."""
    if os.environ.get("POLARS_TEXT_SKIP_POS") == "1":
        pytest.skip("Skipping POS model download for quotation tests")
    df = pl.DataFrame({
        "text": [
            '\u2066@user\u2069 said "hello world" to everyone.',
        ]
    })
    out = df.select(pt.quotation(pl.col("text")).alias("quotes"))
    assert out.height == 1


def test_quotation_null_and_empty() -> None:
    """Null values and empty strings must be handled gracefully."""
    if os.environ.get("POLARS_TEXT_SKIP_POS") == "1":
        pytest.skip("Skipping POS model download for quotation tests")
    df = pl.DataFrame({"text": [None, "", "No quotes here."]})
    out = df.select(pt.quotation(pl.col("text")).alias("quotes"))
    assert out.height == 3
