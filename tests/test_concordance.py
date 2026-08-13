from typing import Any

import polars as pl
import polars_text as pt


def test_concordance_expr_schema() -> None:
    df = pl.DataFrame({"text": ["Hello world, hello again.", None]})
    out = df.select(
        pt.concordance(
            pl.col("text"),
            "hello",
            num_left_tokens=1,
            num_right_tokens=2,
            regex=False,
            case_sensitive=False,
        ).alias("concordance")
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
        pt.concordance(pl.col("text"), "hello", num_left_tokens=1, num_right_tokens=1)
        .list.explode()
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
    expr = pt.concordance(pl.col("text"), "").list.explode().struct.unnest()
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


def _single_concordance(text: str, search_word: str, **kwargs: Any) -> dict[str, object]:
    return (
        pl.DataFrame({"text": [text]})
        .select(pt.concordance(pl.col("text"), search_word, **kwargs).alias("hits"))
        .item()
    )[0]


def test_concordance_legacy_default_counts_punctuation_tokens() -> None:
    hit = _single_concordance(
        "alpha one , , , target . . three omega",
        "target",
        num_left_tokens=2,
        num_right_tokens=2,
    )

    assert hit == {
        "left_context": ", ,",
        "matched_text": "target",
        "right_context": ". .",
        "start_idx": 16,
        "end_idx": 22,
        "l1": ",",
        "r1": ".",
    }


def test_concordance_remove_punct_preserves_raw_context_separators() -> None:
    hit = _single_concordance(
        "alpha one , , , target . . three omega",
        "target",
        num_left_tokens=2,
        num_right_tokens=2,
        remove_punct=True,
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


def test_concordance_remove_punct_uses_unicode_character_offsets() -> None:
    hit = _single_concordance(
        "猫 🐾 前 target 。 後 🐾",
        "target",
        num_left_tokens=1,
        num_right_tokens=1,
        remove_punct=True,
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


def test_concordance_remove_punct_zero_contexts_are_empty() -> None:
    hit = _single_concordance(
        "one , target . two",
        "target",
        num_left_tokens=0,
        num_right_tokens=0,
        remove_punct=True,
    )

    assert (hit["left_context"], hit["right_context"], hit["l1"], hit["r1"]) == (
        "",
        "",
        "",
        "",
    )


def test_concordance_remove_punct_does_not_change_literal_matching() -> None:
    hits = (
        pl.DataFrame({"text": ["hello, world hello world"]})
        .select(
            pt.concordance(
                pl.col("text"),
                "hello world",
                remove_punct=True,
            ).alias("hits")
        )
        .item()
    )

    assert [hit["start_idx"] for hit in hits] == [13]


def test_concordance_remove_punct_does_not_change_regex_matching() -> None:
    hits = (
        pl.DataFrame({"text": ["cat, dog cat dog"]})
        .select(
            pt.concordance(
                pl.col("text"),
                r"cat[ ,]+dog",
                regex=True,
                remove_punct=True,
            ).alias("hits")
        )
        .item()
    )

    assert [hit["matched_text"] for hit in hits] == ["cat, dog", "cat dog"]


def test_concordance_remove_punct_filters_symbol_only_tokens() -> None:
    hit = _single_concordance(
        "left © ™ target ® right",
        "target",
        num_left_tokens=1,
        num_right_tokens=1,
        remove_punct=True,
    )

    assert (hit["left_context"], hit["right_context"], hit["l1"], hit["r1"]) == (
        "left © ™ ",
        " ® right",
        "left",
        "right",
    )
