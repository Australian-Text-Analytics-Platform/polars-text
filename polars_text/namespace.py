from __future__ import annotations

import os
from typing import Literal

import polars as pl

from . import _expressions


@pl.api.register_expr_namespace("text")
class TextNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def tokenize(
        self,
        *,
        model: str,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        cache: str | os.PathLike[str] | None = None,
    ) -> pl.Expr:
        return _expressions.tokenize(
            self._expr,
            model=model,
            lowercase=lowercase,
            remove_punctuation=remove_punctuation,
            cache=cache,
        )

    def concordance(
        self,
        query: str,
        *,
        left_tokens: int = 5,
        right_tokens: int = 5,
        regex: bool = False,
        case_sensitive: bool = False,
        ignore_punctuation: bool = False,
    ) -> pl.Expr:
        return _expressions.concordance(
            self._expr,
            query,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
            regex=regex,
            case_sensitive=case_sensitive,
            ignore_punctuation=ignore_punctuation,
        )

    def clean_text(self) -> pl.Expr:
        return _expressions.clean_text(self._expr)

    def word_count(self) -> pl.Expr:
        return _expressions.word_count(self._expr)

    def char_count(self) -> pl.Expr:
        return _expressions.char_count(self._expr)

    def sentence_count(self) -> pl.Expr:
        return _expressions.sentence_count(self._expr)

    def embedding(
        self,
        *,
        model: str | None = None,
        cache: str | os.PathLike[str] | None = None,
        batch_size: int | None = None,
    ) -> pl.Expr:
        return _expressions.embedding(
            self._expr,
            model=model,
            cache=cache,
            batch_size=batch_size,
        )

    def topic_modeling(
        self,
        *,
        embedding_model: str | None = None,
        embedding_cache: str | os.PathLike[str] | None = None,
        segmentation: Literal["automatic", "line", "sentence"] = "automatic",
        max_tokens: int = 256,
        seed: int = 42,
        min_topic_size: int = 10,
        tokenizer_model: str | None = None,
        lowercase: bool = True,
    ) -> pl.Expr:
        """Cluster a document column and emit one run-level topic struct.

        Output contains separate document outcomes, complete topic metadata,
        segment counts, and an optional projection context.
        """
        return _expressions.topic_modeling(
            self._expr,
            embedding_model=embedding_model,
            embedding_cache=embedding_cache,
            segmentation=segmentation,
            max_tokens=max_tokens,
            seed=seed,
            min_topic_size=min_topic_size,
            tokenizer_model=tokenizer_model,
            lowercase=lowercase,
        )
