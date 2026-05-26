from __future__ import annotations

import os

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from .token_cache import cached_tokenize_expr, uncached_tokenize_expr
from .utils import PLUGIN_PATH


def _tokenize_kwargs(
    *, lowercase: bool, remove_punct: bool, model: str | None
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
    }
    if model is not None:
        kwargs["model_id"] = model
    return kwargs


def tokenize(
    expr: IntoExpr,
    *,
    lowercase: bool = True,
    remove_punct: bool = True,
    model: str | None = None,
    cache: str | os.PathLike[str] | None = None,
) -> pl.Expr:
    """Tokenize and emit a list of ``{token, start, end}`` structs per row.

    ``start`` / ``end`` are character offsets into the (lowercased, if
    ``lowercase=True``) text. If ``cache`` is a path, tokenization results are
    persisted in a DuckDB cache at that location and reused by content hash.
    """
    if cache is None:
        return uncached_tokenize_expr(
            expr,
            lowercase=lowercase,
            remove_punct=remove_punct,
            model=model,
        )
    if not isinstance(expr, pl.Expr):
        expr = pl.col(expr) if isinstance(expr, str) else pl.lit(expr)
    return cached_tokenize_expr(
        expr,
        cache=cache,
        lowercase=lowercase,
        remove_punct=remove_punct,
        model=model,
    )


def concordance(
    expr: IntoExpr,
    search_word: str,
    *,
    num_left_tokens: int = 5,
    num_right_tokens: int = 5,
    regex: bool = False,
    case_sensitive: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="concordance",
        args=expr,
        kwargs={
            "search_word": search_word,
            "num_left_tokens": num_left_tokens,
            "num_right_tokens": num_right_tokens,
            "regex": regex,
            "case_sensitive": case_sensitive,
        },
        is_elementwise=True,
    )


def clean_text(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="clean_text",
        args=expr,
        is_elementwise=True,
    )


def word_count(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="word_count",
        args=expr,
        is_elementwise=True,
    )


def char_count(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="char_count",
        args=expr,
        is_elementwise=True,
    )


def sentence_count(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="sentence_count",
        args=expr,
        is_elementwise=True,
    )


__all__ = [
    "tokenize",
    "concordance",
    "clean_text",
    "word_count",
    "char_count",
    "sentence_count",
]
