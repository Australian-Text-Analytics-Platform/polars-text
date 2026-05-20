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
    model: str | None = None,
) -> pl.Expr:
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
    }
    if model is not None:
        kwargs["model_id"] = model
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="tokenize",
        args=expr,
        kwargs=kwargs,
        is_elementwise=True,
    )


def tokenize_with_offsets(
    expr: IntoExpr,
    *,
    lowercase: bool = True,
    remove_punct: bool = True,
    model: str | None = None,
) -> pl.Expr:
    """Tokenize and emit a list of ``{token, start, end}`` structs per row.

    ``start`` / ``end`` are character offsets into the (lowercased, if
    ``lowercase=True``) text. This is the schema Phase 2 persists as a
    tokens column on derived nodes.
    """
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
    }
    if model is not None:
        kwargs["model_id"] = model
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="tokenize_with_offsets",
        args=expr,
        kwargs=kwargs,
        is_elementwise=True,
    )


def tokenize_with_cache_lookup(
    expr: IntoExpr,
    *,
    user_id: str,
    bucket_filename: str,
    lowercase: bool = True,
    remove_punct: bool = True,
    model: str | None = None,
    require_env_cache_dir: bool = False,
) -> pl.Expr:
    """Lazy on-demand tokenisation with an integrated per-row cache.

    On `.collect()`, looks each row's content hash up in
    `${LDACA_TOKENS_CACHE_DIR}/{user_id}/tokens/{bucket_filename}` plus any
    sibling `__delta__*.parquet` files. Cache hits return the persisted
    `List<Struct<token, start, end>>`; misses fall through to the same
    HF/Jieba/Lindera backend `tokenize_with_offsets` uses and append a
    fresh delta parquet under an advisory `flock`.

    The serialised lazy plan carries only `bucket_filename` + `user_id`
    (both stable across machines) — no absolute paths. That makes
    workspace `.plbin` files portable across installs without the
    repair-pass / sidecar machinery the eager hash-join cache required.

    Set `require_env_cache_dir=True` in production analysis workers so a
    missing `LDACA_TOKENS_CACHE_DIR` env at execution time fails loudly
    instead of silently writing to `/tmp`.
    """
    kwargs: dict[str, object] = {
        "lowercase": lowercase,
        "remove_punct": remove_punct,
        "bucket_filename": bucket_filename,
        "user_id": user_id,
        "require_env_cache_dir": require_env_cache_dir,
    }
    if model is not None:
        kwargs["model_id"] = model
    # Pass the precomputed hash as a second positional input. Rationale:
    # polars' Series.hash uses a feature-gated `PlSeedableRandomStateQuality`
    # whose default-seed contract has shifted between polars versions —
    # computing the hash on the Python side lets the cache map round-trip
    # bit-compatibly with whatever hash function the eager `tokens_cache`
    # path used to populate the same bucket, without coupling the Rust
    # expression to a specific polars internal hash version.
    source = pl.col(expr) if isinstance(expr, str) else expr
    return register_plugin_function(
        plugin_path=PLUGIN_PATH,
        function_name="tokenize_with_cache_lookup",
        args=[source, source.hash()],
        kwargs=kwargs,
        # State is maintained across rows within a batch (the in-memory
        # cache map + the queued-misses buffer), so polars must not split
        # a batch in ways that violate the per-batch invariants. The cache
        # map itself is fine across batches because it's rebuilt from disk
        # each invocation; the miss-buffer flush is per-invocation too.
        is_elementwise=False,
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
    "tokenize_with_offsets",
    "tokenize_with_cache_lookup",
    "concordance",
    "clean_text",
    "word_count",
    "char_count",
    "sentence_count",
]
