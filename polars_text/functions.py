from __future__ import annotations

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from .utils import PLUGIN_PATH


def tokenize(
    expr: IntoExpr,
    *,
    lowercase: bool = True,
    remove_punct: bool = True,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="tokenize",
        args=expr,
        kwargs={"lowercase": lowercase, "remove_punct": remove_punct},
        is_elementwise=True,
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


def quotation(
    expr: IntoExpr,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="quotation",
        args=expr,
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
    "quotation",
    "clean_text",
    "word_count",
    "char_count",
    "sentence_count",
]
