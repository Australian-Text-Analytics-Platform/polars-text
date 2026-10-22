from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

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


def _remote_embedding_kwargs(
    *,
    endpoint: str | None,
    api_key_env: str | None,
    api_key_file: str | os.PathLike[str] | None,
    remote_model: str | None,
    user: str | None,
    batch_size: int | None,
    concurrency: int | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    """Plugin kwargs for remote embedding; empty (local embedding) without an endpoint.

    Keys never enter the plan: only the name of the environment variable or
    the path of the file holding the key does. Errors name the setting, never
    its value.
    """
    others = {
        "embedding_api_key_env": api_key_env,
        "embedding_api_key_file": api_key_file,
        "embedding_remote_model": remote_model,
        "embedding_user": user,
        "embedding_batch_size": batch_size,
        "embedding_concurrency": concurrency,
        "embedding_timeout_s": timeout_s,
    }
    if endpoint is None or not endpoint.strip():
        for name, value in others.items():
            if value is not None:
                raise ValueError(f"{name} needs embedding_endpoint")
        return {}
    _require_feature("remote-embedding", "embedding_endpoint")
    parts = urlsplit(endpoint.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(
            "embedding_endpoint must be an http or https URL such as "
            "https://gateway.example/v1"
        )
    if parts.username is not None or parts.password is not None:
        raise ValueError(
            "embedding_endpoint must not contain credentials; give the key with "
            "embedding_api_key_env or embedding_api_key_file"
        )
    if parts.query or parts.fragment:
        raise ValueError(
            "embedding_endpoint must not have a query or fragment; give the key "
            "with embedding_api_key_env or embedding_api_key_file"
        )
    if api_key_env is not None and api_key_file is not None:
        raise ValueError(
            "set only one of embedding_api_key_env and embedding_api_key_file"
        )
    if api_key_env is not None and not os.environ.get(api_key_env, "").strip():
        raise ValueError(
            "the environment variable named by embedding_api_key_env is not set"
        )
    key_file = None if api_key_file is None else str(Path(api_key_file))
    if key_file is not None and not Path(key_file).is_file():
        raise ValueError("the file named by embedding_api_key_file does not exist")
    if batch_size is not None:
        _positive(batch_size, "embedding_batch_size")
    if concurrency is not None:
        _positive(concurrency, "embedding_concurrency")
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not timeout_s > 0
    ):
        raise ValueError("embedding_timeout_s must be a positive number of seconds")
    values = {
        "embedding_endpoint": endpoint.strip(),
        "embedding_api_key_env": api_key_env,
        "embedding_api_key_file": key_file,
        "embedding_remote_model": _model_id(remote_model),
        "embedding_user": user,
        "embedding_batch_size": batch_size,
        "embedding_concurrency": concurrency,
        "embedding_timeout_s": None if timeout_s is None else float(timeout_s),
    }
    return {name: value for name, value in values.items() if value is not None}


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
        progress_path: str | os.PathLike[str] | None = None,
    ) -> pl.Expr:
        """Tokenise each document.

        With ``progress_path``, the documents tokenised so far are written to
        that JSON file (``{"step", "steps", "label", "done", "total", "unit",
        "updated_at_ms"}``, at most twice a second); every chunk of the column
        adds to the same count, and ``total`` is left to the caller.
        """
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
                "progress_path": _cache_path(progress_path),
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
        embedding_endpoint: str | None = None,
        embedding_api_key_env: str | None = None,
        embedding_api_key_file: str | os.PathLike[str] | None = None,
        embedding_remote_model: str | None = None,
        embedding_user: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_concurrency: int | None = None,
        embedding_timeout_s: float | None = None,
    ) -> pl.Expr:
        """Sentence embeddings of each text (or each text in a list).

        With ``embedding_endpoint`` (an OpenAI-compatible API root such as
        ``https://gateway.example/v1``) the vectors come from that server's
        ``/embeddings`` instead of the local ONNX model; ``model`` still names
        the model whose tokenizer, length limit and normalisation apply. See
        the README section "Remote embeddings".
        """
        _require_feature("embedding", "embedding")
        if batch_size is not None:
            _positive(batch_size, "batch_size")
        remote = _remote_embedding_kwargs(
            endpoint=embedding_endpoint,
            api_key_env=embedding_api_key_env,
            api_key_file=embedding_api_key_file,
            remote_model=embedding_remote_model,
            user=embedding_user,
            batch_size=embedding_batch_size,
            concurrency=embedding_concurrency,
            timeout_s=embedding_timeout_s,
        )
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="embedding",
            args=self._expr,
            kwargs={
                "embedder_model": _model_id(model),
                "cache": _cache_path(cache),
                "batch_size": batch_size,
                **remote,
            },
            is_elementwise=True,
        )

    def segment_token_counts(
        self,
        unit: Literal["automatic", "paragraph", "line", "sentence"] = "automatic",
        *,
        model: str | None = None,
    ) -> pl.Expr:
        """Token counts of each row's units, as Topic Segments are sized.

        ``unit`` follows the segmentation modes: ``automatic`` measures the
        whole text, ``paragraph`` the paragraphs Automatic packs, ``line``
        non-empty lines and ``sentence`` UAX #29 sentences. Counts use the
        embedding model's tokenizer, include its special tokens and never
        split a unit; units without letters or digits are skipped. Returns
        ``List(UInt32)``; null rows stay null. Loads only the tokenizer.
        """
        _require_feature("topic-modeling", "segment_token_counts")
        if unit not in {"automatic", "paragraph", "line", "sentence"}:
            raise ValueError(
                "unit must be 'automatic', 'paragraph', 'line' or 'sentence'"
            )
        return register_plugin_function(
            plugin_path=PLUGIN_PATH,
            function_name="segment_token_counts",
            args=self._expr,
            kwargs={"embedder_model": _model_id(model), "unit": unit},
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
        max_topic_size: int | None = None,
        tokenizer_model: str | None = None,
        lowercase: bool = True,
        cluster_sample_size: int | None = None,
        progress_path: str | os.PathLike[str] | None = None,
        embedding_endpoint: str | None = None,
        embedding_api_key_env: str | None = None,
        embedding_api_key_file: str | os.PathLike[str] | None = None,
        embedding_remote_model: str | None = None,
        embedding_user: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_concurrency: int | None = None,
        embedding_timeout_s: float | None = None,
    ) -> pl.Expr:
        """Cluster a document column and emit one run-level topic struct.

        ``segmentation="automatic"`` packs neighbouring paragraphs into one
        segment while they fit ``max_tokens``; ``"line"`` keeps one line per
        segment and ``"sentence"`` one sentence per segment.

        With ``cluster_sample_size`` and more segments than that, a seeded
        sample of that many segments is clustered and every other segment
        takes its nearest sampled segment's topic; ``clustered_segments`` in
        the output reports the sample size (null when all were clustered).

        With ``max_topic_size=None`` (Auto), a topic that is the main topic of
        more than half of the documents is split with a cap below its size
        when the split keeps most of its segments in topics; the output's
        ``auto_decision`` ("not_needed", "split" or "kept") and
        ``auto_document_share`` (that topic's share of documents before Auto)
        explain the outcome; ``largest_topic_size`` is the segment count of
        the largest topic, the scale a fixed ``max_topic_size`` works on.

        With ``progress_path``, progress is written to that JSON file in five
        steps: ``segmenting`` (documents), ``embedding`` (segments, cached
        ones counted first), ``arranging``, ``grouping`` and ``topic_words``
        (the last three without counts).

        With ``embedding_endpoint``, segment embeddings come from that
        OpenAI-compatible server instead of the local model (see
        ``embedding``); segmentation, clustering and topic words stay local.
        A failed remote embedding fails the run: it never falls back to the
        local model.

        Output contains separate document outcomes, complete topic metadata,
        segment counts, and an optional projection context.
        """
        _require_feature("topic-modeling", "topic_modeling")
        if segmentation not in {"automatic", "line", "sentence"}:
            raise ValueError("segmentation must be 'automatic', 'line', or 'sentence'")
        _positive(max_tokens, "max_tokens")
        _positive(min_topic_size, "min_topic_size", minimum=2)
        if max_topic_size is not None:
            # None means Auto: the native pipeline caps topics at a share of all segments.
            _positive(max_topic_size, "max_topic_size", minimum=min_topic_size + 1)
        if cluster_sample_size is not None:
            _positive(cluster_sample_size, "cluster_sample_size", minimum=min_topic_size)
        remote = _remote_embedding_kwargs(
            endpoint=embedding_endpoint,
            api_key_env=embedding_api_key_env,
            api_key_file=embedding_api_key_file,
            remote_model=embedding_remote_model,
            user=embedding_user,
            batch_size=embedding_batch_size,
            concurrency=embedding_concurrency,
            timeout_s=embedding_timeout_s,
        )

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
                "max_cluster_size": max_topic_size,
                "vectorizer_model": _model_id(tokenizer_model),
                "lowercase": lowercase,
                "cluster_sample_size": cluster_sample_size,
                "progress_path": _cache_path(progress_path),
                **remote,
            },
            is_elementwise=False,
            returns_scalar=True,
        )
