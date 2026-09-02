from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

from . import _internal
from ._internal import compiled_features

PLUGIN_PATH = Path(_internal.__file__).resolve()

def _require_feature(feature: str, operation: str) -> None:
    if feature not in compiled_features():
        raise RuntimeError(
            f"{operation} requires the '{feature}' feature; rebuild polars-text "
            "with that feature or install the default full wheel"
        )


def _model_id(model: str | None, *, required_by: str | None = None) -> str | None:
    if model is None:
        if required_by is not None:
            raise ValueError(f"{required_by} requires an explicit model ID")
        return None
    normalized = model.strip()
    if not normalized:
        if required_by is not None:
            raise ValueError(f"{required_by} requires an explicit model ID")
        return None
    return normalized


def _cache_path(cache: str | os.PathLike[str] | None) -> str | None:
    return str(Path(cache)) if cache is not None else None


def _positive(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _non_negative(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


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
        _require_feature("tokenization", "tokenize")
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="tokenize",
            args=self._expr,
            kwargs={
                "lowercase": lowercase,
                "remove_punct": remove_punctuation,
                "model_id": _model_id(model, required_by="tokenize"),
                "cache": _cache_path(cache),
            },
            is_elementwise=True,
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
        _require_feature("tokenization", "concordance")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="concordance",
            args=self._expr,
            kwargs={
                "search_word": query,
                "num_left_tokens": _non_negative(left_tokens, "left_tokens"),
                "num_right_tokens": _non_negative(right_tokens, "right_tokens"),
                "regex": regex,
                "case_sensitive": case_sensitive,
                "ignore_punctuation": ignore_punctuation,
            },
            is_elementwise=True,
        )

    def clean_text(self) -> pl.Expr:
        return (
            self._expr.cast(pl.String)
            .fill_null("")
            .str.to_lowercase()
            .str.replace_all(r"[[:punct:]0-9]+", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )

    def word_count(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="word_count",
            args=self._expr,
            is_elementwise=True,
        )

    def char_count(self) -> pl.Expr:
        return (
            self._expr.cast(pl.String).str.len_chars().fill_null(0).cast(pl.Int64)
        )

    def sentence_count(self) -> pl.Expr:
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="sentence_count",
            args=self._expr,
            is_elementwise=True,
        )

    def embedding(
        self,
        *,
        model: str | None = None,
        cache: str | os.PathLike[str] | None = None,
        batch_size: int | None = None,
    ) -> pl.Expr:
        _require_feature("embedding", "embedding")
        if batch_size is not None:
            _positive(batch_size, "batch_size")
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="embedding",
            args=self._expr,
            kwargs={
                "embedder_model": _model_id(model),
                "cache": _cache_path(cache),
                "batch_size": batch_size,
            },
            is_elementwise=True,
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
        _require_feature("topic-modeling", "topic_modeling")
        if segmentation not in {"automatic", "line", "sentence"}:
            raise ValueError("segmentation must be 'automatic', 'line', or 'sentence'")
        _positive(max_tokens, "max_tokens")
        _positive(min_topic_size, "min_topic_size", minimum=2)

        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="topic_modeling",
            args=self._expr,
            kwargs={
                "embedder_model": _model_id(embedding_model),
                "cache": _cache_path(embedding_cache),
                "segmentation_method": segmentation,
                "max_tokens": max_tokens,
                "seed": seed,
                "min_cluster_size": min_topic_size,
                "vectorizer_model": _model_id(tokenizer_model),
                "lowercase": lowercase,
            },
            is_elementwise=False,
            returns_scalar=True,
        )
