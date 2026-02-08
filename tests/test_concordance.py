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
        pl.Struct([
            pl.Field("left_context", pl.String),
            pl.Field("matched_text", pl.String),
            pl.Field("right_context", pl.String),
            pl.Field("start_idx", pl.Int64),
            pl.Field("end_idx", pl.Int64),
            pl.Field("l1", pl.String),
            pl.Field("r1", pl.String),
        ])
    )


def test_concordance_namespace_explode_unnest() -> None:
    df = pl.DataFrame({"text": ["Hello world, hello again."]})
    expr = (
        pl
        .col("text")
        .text.concordance("hello", num_left_tokens=1, num_right_tokens=1)
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
