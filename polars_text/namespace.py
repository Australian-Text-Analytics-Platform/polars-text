from __future__ import annotations

import os
from typing import Literal

import polars as pl

from . import functions


@pl.api.register_expr_namespace("text")
class TextNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def tokenize(
        self,
        *,
        model: str,
        lowercase: bool = True,
        remove_punct: bool = True,
        cache: str | os.PathLike[str] | None = None,
    ) -> pl.Expr:
        return functions.tokenize(
            self._expr,
            model=model,
            lowercase=lowercase,
            remove_punct=remove_punct,
            cache=cache,
        )

    def concordance(
        self,
        search_word: str,
        *,
        num_left_tokens: int = 5,
        num_right_tokens: int = 5,
        regex: bool = False,
        case_sensitive: bool = False,
        remove_punct: bool = False,
    ) -> pl.Expr:
        return functions.concordance(
            self._expr,
            search_word,
            num_left_tokens=num_left_tokens,
            num_right_tokens=num_right_tokens,
            regex=regex,
            case_sensitive=case_sensitive,
            remove_punct=remove_punct,
        )

    def clean_text(self) -> pl.Expr:
        return functions.clean_text(self._expr)

    def word_count(self) -> pl.Expr:
        return functions.word_count(self._expr)

    def char_count(self) -> pl.Expr:
        return functions.char_count(self._expr)

    def sentence_count(self) -> pl.Expr:
        return functions.sentence_count(self._expr)

    def embedding(
        self,
        *,
        embedder_model: str | None = None,
        cache: str | os.PathLike[str] | None = None,
        batch_size: int | None = None,
    ) -> pl.Expr:
        return functions.embedding(
            self._expr,
            embedder_model=embedder_model,
            cache=cache,
            batch_size=batch_size,
        )

    def topic_modeling(
        self,
        *,
        embedder_model: str | None = None,
        cache: str | os.PathLike[str] | None = None,
        segmentation_method: Literal["automatic", "paragraph", "sentence"] = "automatic",
        max_tokens: int = 256,
        overlap: int = 32,
        reduce_dims: int = 5,
        seed: int = 42,
        min_cluster_size: int = 10,
        min_samples: int | None = None,
        vectorizer_model: str | None = None,
        lowercase: bool = True,
    ) -> pl.Expr:
        """Cluster a document column and emit one run-level topic struct.

        Output contains separate document outcomes and complete topic metadata,
        plus segment counts, truncation reporting, and native stage timings.
        """
        return functions.topic_modeling(
            self._expr,
            embedder_model=embedder_model,
            cache=cache,
            segmentation_method=segmentation_method,
            max_tokens=max_tokens,
            overlap=overlap,
            reduce_dims=reduce_dims,
            seed=seed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            vectorizer_model=vectorizer_model,
            lowercase=lowercase,
        )
