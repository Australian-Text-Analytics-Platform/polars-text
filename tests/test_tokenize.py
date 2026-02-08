import polars as pl
import polars_text as pt


def test_tokenize_expr() -> None:
    df = pl.DataFrame({"text": ["Hello, world!", None]})
    out = df.select(pt.tokenize(pl.col("text")))
    assert out.shape == (2, 1)


def test_tokenize_namespace() -> None:
    df = pl.DataFrame({"text": ["Hello, world!", None]})
    out = df.select(pl.col("text").text.tokenize())
    assert out.shape == (2, 1)
