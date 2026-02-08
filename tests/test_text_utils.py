import polars as pl
import polars_text as pt


def test_clean_text() -> None:
    series = pl.Series("text", ["Hello, World! 123", None])
    out = pl.DataFrame({"text": series}).select(
        pt.clean_text(pl.col("text")).alias("clean")
    )
    assert out["clean"][0] == "hello world"
    assert out["clean"][1] == ""


def test_word_count() -> None:
    df = pl.DataFrame({"text": ["hello world", "  one   two  ", None]})
    out = df.select(pl.col("text").text.word_count().alias("wc"))
    assert out["wc"].to_list() == [2, 2, 0]


def test_char_count() -> None:
    df = pl.DataFrame({"text": ["abc", "", None]})
    out = df.select(pl.col("text").text.char_count().alias("cc"))
    assert out["cc"].to_list() == [3, 0, 0]


def test_sentence_count() -> None:
    df = pl.DataFrame({"text": ["One. Two? Three!", "", None]})
    out = df.select(pl.col("text").text.sentence_count().alias("sc"))
    assert out["sc"].to_list() == [3, 0, 0]
