from typing import Any, cast

import polars as pl
import polars_text  # noqa: F401


def test_concordance_expr_schema() -> None:
    df = pl.DataFrame({"text": ["Hello world, hello again.", None]})
    out = df.select(
        cast(Any, pl.col("text"))
        .text.concordance(
            "hello",
            left_tokens=1,
            right_tokens=2,
            regex=False,
            case_sensitive=False,
        )
        .alias("concordance")
    )
    dtype = out.schema["concordance"]
    assert dtype == pl.List(
        pl.Struct(
            [
                pl.Field("left_context", pl.String),
                pl.Field("matched_text", pl.String),
                pl.Field("right_context", pl.String),
                pl.Field("start_idx", pl.Int64),
                pl.Field("end_idx", pl.Int64),
                pl.Field("l1", pl.String),
                pl.Field("r1", pl.String),
            ]
        )
    )


def test_concordance_namespace_explode_unnest() -> None:
    df = pl.DataFrame({"text": ["Hello world, hello again."]})
    expr = (
        cast(Any, pl.col("text"))
        .text.concordance("hello", left_tokens=1, right_tokens=1)
        .list.explode(empty_as_null=True)
        .struct.unnest()
    )
    out = df.select(expr)
    assert out.height == 2
    assert out.columns == [
        "left_context",
        "matched_text",
        "right_context",
        "start_idx",
        "end_idx",
        "l1",
        "r1",
    ]


def test_concordance_empty_search_explode_unnest() -> None:
    df = pl.DataFrame({"text": ["Hello world."]})
    expr = (
        cast(Any, pl.col("text"))
        .text.concordance("")
        .list.explode(empty_as_null=True)
        .struct.unnest()
    )
    out = df.select(expr)
    assert out.height == 1
    assert out.columns == [
        "left_context",
        "matched_text",
        "right_context",
        "start_idx",
        "end_idx",
        "l1",
        "r1",
    ]
    assert out.null_count().to_dicts()[0] == {
        "left_context": 1,
        "matched_text": 1,
        "right_context": 1,
        "start_idx": 1,
        "end_idx": 1,
        "l1": 1,
        "r1": 1,
    }


def _single_concordance(
    text: str, search_word: str, **kwargs: Any
) -> dict[str, object]:
    return (
        pl.DataFrame({"text": [text]})
        .select(cast(Any, pl.col("text")).text.concordance(search_word, **kwargs).alias("hits"))
        .item()
    )[0]


def test_concordance_default_counts_punctuation_and_preserves_source() -> None:
    hit = _single_concordance(
        "alpha one , , , target . . three omega",
        "target",
        left_tokens=2,
        right_tokens=2,
    )

    assert hit == {
        "left_context": ", , ",
        "matched_text": "target",
        "right_context": " . .",
        "start_idx": 16,
        "end_idx": 22,
        "l1": ",",
        "r1": ".",
    }


def test_concordance_ignore_punctuation_preserves_raw_context_separators() -> None:
    hit = _single_concordance(
        "alpha one , , , target . . three omega",
        "target",
        left_tokens=2,
        right_tokens=2,
        ignore_punctuation=True,
    )

    assert hit == {
        "left_context": "alpha one , , , ",
        "matched_text": "target",
        "right_context": " . . three omega",
        "start_idx": 16,
        "end_idx": 22,
        "l1": "one",
        "r1": "three",
    }


def test_concordance_ignore_punctuation_uses_unicode_character_offsets() -> None:
    hit = _single_concordance(
        "猫 🐾 前 target 。 後 🐾",
        "target",
        left_tokens=1,
        right_tokens=1,
        ignore_punctuation=True,
    )

    assert hit == {
        "left_context": "前 ",
        "matched_text": "target",
        "right_context": " 。 後",
        "start_idx": 6,
        "end_idx": 12,
        "l1": "前",
        "r1": "後",
    }


def test_concordance_ignore_punctuation_zero_contexts_are_empty() -> None:
    hit = _single_concordance(
        "one , target . two",
        "target",
        left_tokens=0,
        right_tokens=0,
        ignore_punctuation=True,
    )

    assert (hit["left_context"], hit["right_context"], hit["l1"], hit["r1"]) == (
        "",
        "",
        "",
        "",
    )


def test_concordance_ignore_punctuation_does_not_change_literal_matching() -> None:
    hits = (
        pl.DataFrame({"text": ["hello, world hello world"]})
        .select(
            cast(Any, pl.col("text"))
            .text.concordance(
                "hello world",
                ignore_punctuation=True,
            )
            .alias("hits")
        )
        .item()
    )

    assert [hit["start_idx"] for hit in hits] == [13]


def test_concordance_ignore_punctuation_does_not_change_regex_matching() -> None:
    hits = (
        pl.DataFrame({"text": ["cat, dog cat dog"]})
        .select(
            cast(Any, pl.col("text"))
            .text.concordance(
                r"cat[ ,]+dog",
                regex=True,
                ignore_punctuation=True,
            )
            .alias("hits")
        )
        .item()
    )

    assert [hit["matched_text"] for hit in hits] == ["cat, dog", "cat dog"]


def test_concordance_ignore_punctuation_filters_symbol_only_tokens() -> None:
    hit = _single_concordance(
        "left © ™ target ® right",
        "target",
        left_tokens=1,
        right_tokens=1,
        ignore_punctuation=True,
    )

    assert (hit["left_context"], hit["right_context"], hit["l1"], hit["r1"]) == (
        "left © ™ ",
        " ® right",
        "left",
        "right",
    )


def test_concordance_partial_token_match_preserves_fragment_windows() -> None:
    hit = _single_concordance(
        "XXX concatenate YYY",
        "cat",
        left_tokens=1,
        right_tokens=1,
        regex=True,
    )

    assert hit == {
        "left_context": "con",
        "matched_text": "cat",
        "right_context": "enate",
        "start_idx": 7,
        "end_idx": 10,
        "l1": "con",
        "r1": "enate",
    }


def test_concordance_literal_partial_token_match_preserves_fragments() -> None:
    hit = _single_concordance(
        "XXX concatenate YYY", "cat", left_tokens=1, right_tokens=1
    )
    assert (hit["l1"], hit["matched_text"], hit["r1"]) == ("con", "cat", "enate")


def test_concordance_many_hits_remain_ordered_with_unicode_offsets() -> None:
    hits = (
        pl.DataFrame({"text": ["猫 cat cat cat 🐾 cat"]})
        .select(cast(Any, pl.col("text")).text.concordance("cat").alias("hits"))
        .item()
    )
    assert [hit["start_idx"] for hit in hits] == [2, 6, 10, 16]
    assert [hit["matched_text"] for hit in hits] == ["cat"] * 4


def test_concordance_zero_width_matches_are_source_ordered() -> None:
    hits = (
        pl.DataFrame({"text": ["ab"]})
        .select(cast(Any, pl.col("text")).text.concordance(r"^", regex=True).alias("hits"))
        .item()
    )
    assert [(hit["start_idx"], hit["end_idx"]) for hit in hits] == [(0, 0)]
