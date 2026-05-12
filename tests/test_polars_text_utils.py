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
    out = df.select(pt.word_count(pl.col("text")).alias("wc"))
    assert out["wc"].to_list() == [2, 2, 0]


def test_char_count() -> None:
    df = pl.DataFrame({"text": ["abc", "", None]})
    out = df.select(pt.char_count(pl.col("text")).alias("cc"))
    assert out["cc"].to_list() == [3, 0, 0]


def test_sentence_count() -> None:
    df = pl.DataFrame({"text": ["One. Two? Three!", "", None]})
    out = df.select(pt.sentence_count(pl.col("text")).alias("sc"))
    assert out["sc"].to_list() == [3, 0, 0]


# ---------------------------------------------------------------------------
# Phase 3.3 + 3.4: CJK / Unicode-aware counts. English is byte-identical.
# ---------------------------------------------------------------------------


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
    out = df.select(pt.sentence_count(pl.col("text")).alias("sc"))
    assert out["sc"].to_list() == [3, 2, 2]


def test_word_count_pure_cjk_counts_each_character() -> None:
    df = pl.DataFrame(
        {
            "text": [
                "今天天气很好",   # 6 Han chars
                "你好",            # 2 Han chars
                "ご飯",            # 2 Hiragana chars
                "안녕하세요",      # 5 Hangul syllables
            ]
        }
    )
    out = df.select(pt.word_count(pl.col("text")).alias("wc"))
    assert out["wc"].to_list() == [6, 2, 2, 5]


def test_word_count_english_unchanged() -> None:
    """Pre-existing English behaviour is byte-identical — the new heuristic
    only kicks in for pure-CJK text with no whitespace."""
    df = pl.DataFrame(
        {"text": ["hello world", "  one   two  ", "single", "", None]}
    )
    out = df.select(pt.word_count(pl.col("text")).alias("wc"))
    assert out["wc"].to_list() == [2, 2, 1, 0, 0]


def test_word_count_mixed_cjk_and_whitespace_uses_whitespace_split() -> None:
    """Text with internal whitespace falls through to ``split_whitespace``
    even if some tokens contain CJK chars — that's the existing semantic and
    a real user can still get word-level CJK counts by running Tokenise."""
    df = pl.DataFrame({"text": ["Hello 你好", "今天 nice 天气"]})
    out = df.select(pt.word_count(pl.col("text")).alias("wc"))
    assert out["wc"].to_list() == [2, 3]
