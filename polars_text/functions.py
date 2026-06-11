from __future__ import annotations

import os

import polars as pl
from polars._typing import IntoExpr
from polars.plugins import register_plugin_function

from .token_cache import cached_tokenize_expr, uncached_tokenize_expr
from .utils import PLUGIN_PATH


def _normalise_model(model: str | None) -> str:
    if model is None or not model.strip():
        raise ValueError("tokenize requires an explicit tokenizer model ID")
    return model.strip()


def _tokenize_kwargs(
    *, lowercase: bool, remove_punct: bool, model: str
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
        "model_id": model,
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
    model_id = _normalise_model(model)
    if cache is None:
        return uncached_tokenize_expr(
            expr,
            lowercase=lowercase,
            remove_punct=remove_punct,
            model=model_id,
        )
    if not isinstance(expr, pl.Expr):
        expr = pl.col(expr) if isinstance(expr, str) else pl.lit(expr)
    return cached_tokenize_expr(
        expr,
        cache=cache,
        lowercase=lowercase,
        remove_punct=remove_punct,
        model=model_id,
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


def topic_modeling(
    expr: IntoExpr,
    *,
    embedder_model: str | None = None,
    max_tokens: int = 256,
    overlap: int = 32,
    reduce_dims: int = 5,
    seed: int = 42,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    top_k: int = 10,
    vectorizer_model: str | None = None,
    lowercase: bool = True,
    stopwords: list[str] | None = None,
) -> pl.Expr:
    """Cluster a whole document column into topics, one struct emitted per row.

    Unlike the other ``.text`` helpers this is **not** elementwise: clustering
    needs every document at once, so the expression consumes the entire column
    and returns a per-row struct that lines up 1:1 with the input rows:

    ``{dominant_topic: i32, topic_distribution: list[{topic_id, proportion}],
    representative_words: list[str], x: f32, y: f32, n_topics: u32,
    n_chunks: u32}``

    Topic-level fields (``representative_words``/``x``/``y``) are replicated onto
    every row under its dominant topic, and ``n_topics``/``n_chunks`` are global
    counts replicated per row, so callers can recover the bubble chart and
    per-corpus sizes with a plain ``group_by('dominant_topic')`` without any
    extra orchestration. Outlier rows (``dominant_topic == -1``) get an empty
    ``representative_words`` list and origin coords.

    The topic count is whatever HDBSCAN yields for ``min_cluster_size`` (the only
    native topic-count control); there is no post-fit merge to a requested count.

    Pool multiple corpora by concatenating their columns into one before calling
    this, then split the per-row output by your own corpus-index column.
    """
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="topic_modeling",
        args=expr,
        kwargs={
            "embedder_model": embedder_model,
            "max_tokens": max_tokens,
            "overlap": overlap,
            "reduce_dims": reduce_dims,
            "seed": seed,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "top_k": top_k,
            "vectorizer_model": vectorizer_model,
            "lowercase": lowercase,
            "stopwords": stopwords,
        },
        is_elementwise=False,
    )


__all__ = [
    "tokenize",
    "concordance",
    "clean_text",
    "word_count",
    "char_count",
    "sentence_count",
    "topic_modeling",
]
