from __future__ import annotations

import polars as pl

from . import functions


@pl.api.register_expr_namespace("text")
class TextNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def tokenize(self, *, lowercase: bool = True, remove_punct: bool = True) -> pl.Expr:
        return functions.tokenize(
            self._expr, lowercase=lowercase, remove_punct=remove_punct
        )

    def concordance(
        self,
        search_word: str,
        *,
        num_left_tokens: int = 5,
        num_right_tokens: int = 5,
        regex: bool = False,
        case_sensitive: bool = False,
    ) -> pl.Expr:
        return functions.concordance(
            self._expr,
            search_word,
            num_left_tokens=num_left_tokens,
            num_right_tokens=num_right_tokens,
            regex=regex,
            case_sensitive=case_sensitive,
        )

    def quotation(
        self,
    ) -> pl.Expr:
        return functions.quotation(
            self._expr,
        )

    def clean_text(self) -> pl.Expr:
        return functions.clean_text(self._expr)

    def word_count(self) -> pl.Expr:
        return functions.word_count(self._expr)

    def char_count(self) -> pl.Expr:
        return functions.char_count(self._expr)

    def sentence_count(self) -> pl.Expr:
        return functions.sentence_count(self._expr)
