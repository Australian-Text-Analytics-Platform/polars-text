from typing import Any, cast

import polars as pl
import polars_text  # noqa: F401


def test_clean_text() -> None:
    series = pl.Series("text", ["Hello, World! 123", None])
    out = pl.DataFrame({"text": series}).select(
        cast(Any, pl.col("text")).text.clean_text().alias("clean")
    )
    assert out["clean"][0] == "hello world"
    assert out["clean"][1] == ""


def test_word_count() -> None:
    df = pl.DataFrame({"text": ["hello world", "  one   two  ", None]})
    out = df.select(cast(Any, pl.col("text")).text.word_count().alias("wc"))
    assert out["wc"].to_list() == [2, 2, 0]


def test_char_count() -> None:
    df = pl.DataFrame({"text": ["abc", "", None]})
    out = df.select(cast(Any, pl.col("text")).text.char_count().alias("cc"))
    assert out["cc"].to_list() == [3, 0, 0]


def test_sentence_count() -> None:
    df = pl.DataFrame({"text": ["One. Two? Three!", "", None]})
    out = df.select(cast(Any, pl.col("text")).text.sentence_count().alias("sc"))
    assert out["sc"].to_list() == [3, 0, 0]


def test_sentence_count_handles_cjk_terminators() -> None:
    df = pl.DataFrame(
        {
            "text": [
                "今天天气很好。明天也会很好！你想去哪里？",
                "ご飯を食べました。映画を見ますか？",
                "Hello. 你好。",  # mixed EN + ZH terminators
            ]
        }
    )
    out = df.select(cast(Any, pl.col("text")).text.sentence_count().alias("sc"))
    assert out["sc"].to_list() == [3, 2, 2]


def test_word_count_uses_uax29_for_cjk() -> None:
    df = pl.DataFrame(
        {
            "text": [
                "今天天气很好",  # 6 Han chars
                "你好",  # 2 Han chars
                "ご飯",  # 2 Hiragana chars
                "안녕하세요",  # 5 Hangul syllables
            ]
        }
    )
    out = df.select(cast(Any, pl.col("text")).text.word_count().alias("wc"))
    assert out["wc"].to_list() == [6, 2, 2, 1]


def test_word_count_handles_english_nulls_and_empty_strings() -> None:
    df = pl.DataFrame({"text": ["hello world", "  one   two  ", "single", "", None]})
    out = df.select(cast(Any, pl.col("text")).text.word_count().alias("wc"))
    assert out["wc"].to_list() == [2, 2, 1, 0, 0]


def test_word_count_handles_multilingual_text() -> None:
    df = pl.DataFrame({"text": ["Hello 你好", "今天 nice 天气"]})
    out = df.select(cast(Any, pl.col("text")).text.word_count().alias("wc"))
    assert out["wc"].to_list() == [3, 5]


def test_word_count_handles_contractions_and_numbers() -> None:
    df = pl.DataFrame({"text": ["don't stop", "3.14 and 1,000"]})
    out = df.select(cast(Any, pl.col("text")).text.word_count().alias("wc"))
    assert out["wc"].to_list() == [2, 3]
