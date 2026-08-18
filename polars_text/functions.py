from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from ._internal import compiled_features
from .utils import PLUGIN_PATH


def _require_feature(feature: str, operation: str) -> None:
    if feature not in compiled_features():
        raise RuntimeError(
            f"{operation} requires the '{feature}' feature; rebuild polars-text "
            "with that feature or install the default full wheel"
        )


def _normalise_model(model: str | None) -> str:
    if model is None or not model.strip():
        raise ValueError("tokenize requires an explicit tokenizer model ID")
    return model.strip()


def _tokenize_kwargs(
    *,
    lowercase: bool,
    remove_punct: bool,
    model: str,
    cache: str | os.PathLike[str] | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
        "model_id": model,
        "cache": str(Path(cache)) if cache is not None else None,
    }
    return kwargs


def tokenize(
    expr: IntoExpr,
    *,
    model: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    cache: str | os.PathLike[str] | None = None,
) -> pl.Expr:
    """Tokenize and emit a list of ``{token, start, end}`` structs per row.

    ``start`` / ``end`` are character offsets into the (lowercased, if
    ``lowercase=True``) text. If ``cache`` is a path, tokenization results are
    persisted in a DuckDB cache at that location and reused by content hash.
    """
    _require_feature("tokenization", "tokenize")
    model_id = _normalise_model(model)
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="tokenize",
        args=expr,
        kwargs=_tokenize_kwargs(
            lowercase=lowercase,
            remove_punct=remove_punct,
            model=model_id,
            cache=cache,
        ),
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
    remove_punct: bool = False,
) -> pl.Expr:
    _require_feature("tokenization", "concordance")
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
            "remove_punct": remove_punct,
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


def embedding(
    expr: IntoExpr,
    *,
    embedder_model: str | None = None,
    cache: str | os.PathLike[str] | None = None,
    batch_size: int | None = None,
) -> pl.Expr:
    """Embed string or list-of-string values with an ONNX sentence model.

    The Rust plugin owns model download/loading and only accepts Hugging Face
    repositories that publish ONNX artifacts. If ``cache`` is provided, vectors
    are persisted in the DuckDB file at that path and reused by text hash.
    """
    _require_feature("embedding", "embedding")
    kwargs: dict[str, object] = {
        "embedder_model": embedder_model,
        "cache": str(Path(cache)) if cache is not None else None,
        "batch_size": batch_size,
    }
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="embedding",
        args=expr,
        kwargs=kwargs,
        is_elementwise=True,
    )


def topic_modeling(
    expr: IntoExpr,
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
    """Cluster a whole document column and emit one run-level result struct.

    Unlike the other ``.text`` helpers this is **not** elementwise: clustering
    needs every document at once, so the expression consumes the entire column
    and returns exactly one struct:

    ``{documents: list[{doc_index, dominant_topic, topic_distribution}],
    topics: list[{id, representative_words, x, y}], n_chunks,
    truncated_segment_count, stage_timings_ms, clustering_context}``

    The complete topic list is independent of document dominance, so metadata
    remains available for topics that occur in distributions but never dominate
    a document. Document proportions are weighted by retained Topic Segment
    Unicode-character length; clustering itself remains one observation per
    segment.

    ``min_cluster_size`` controls the natural HDBSCAN leaves in this low-level
    API. The opaque clustering context can project those real Topics downward
    without rerunning model stages; Wordflow deliberately fixes this argument
    internally and exposes only the Result-time cluster count.

    Pool multiple corpora by concatenating their columns into one before calling
    this, then split ``documents`` by your own corpus-index mapping.
    """
    _require_feature("topic-modeling", "topic_modeling")
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="topic_modeling",
        args=expr,
        kwargs={
            "embedder_model": embedder_model,
            "cache": str(Path(cache)) if cache is not None else None,
            "segmentation_method": segmentation_method,
            "max_tokens": max_tokens,
            "overlap": overlap,
            "reduce_dims": reduce_dims,
            "seed": seed,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "vectorizer_model": vectorizer_model,
            "lowercase": lowercase,
        },
        is_elementwise=False,
        returns_scalar=True,
    )


__all__ = [
    "tokenize",
    "concordance",
    "clean_text",
    "word_count",
    "char_count",
    "sentence_count",
    "embedding",
    "topic_modeling",
    "compiled_features",
]
