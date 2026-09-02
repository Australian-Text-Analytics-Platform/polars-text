from __future__ import annotations

import inspect
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import Any, cast

import polars as pl
import polars_text
import polars_text.namespace as namespace_module
import pytest
from polars_text.namespace import PLUGIN_PATH, TextNamespace


def test_expression_operations_are_namespace_only() -> None:
    removed = {
        "tokenize",
        "concordance",
        "clean_text",
        "word_count",
        "char_count",
        "sentence_count",
        "embedding",
        "topic_modeling",
        "compiled_features",
    }
    assert removed.isdisjoint(polars_text.__all__)
    assert all(not hasattr(polars_text, name) for name in removed)
    with pytest.raises(ImportError):
        __import__("polars_text.functions")


def test_namespace_signatures_match_the_public_contract() -> None:
    assert str(inspect.signature(TextNamespace.tokenize)) == (
        "(self, *, model: 'str', lowercase: 'bool' = True, "
        "remove_punctuation: 'bool' = True, cache: 'str | os.PathLike[str] | None' = None) "
        "-> 'pl.Expr'"
    )
    assert str(inspect.signature(TextNamespace.concordance)) == (
        "(self, query: 'str', *, left_tokens: 'int' = 5, right_tokens: 'int' = 5, "
        "regex: 'bool' = False, case_sensitive: 'bool' = False, "
        "ignore_punctuation: 'bool' = False) -> 'pl.Expr'"
    )
    assert str(inspect.signature(TextNamespace.topic_modeling)) == (
        "(self, *, embedding_model: 'str | None' = None, "
        "embedding_cache: 'str | os.PathLike[str] | None' = None, "
        "segmentation: \"Literal['automatic', 'line', 'sentence']\" = 'automatic', "
        "max_tokens: 'int' = 256, seed: 'int' = 42, "
        "min_topic_size: 'int' = 10, tokenizer_model: 'str | None' = None, "
        "lowercase: 'bool' = True) -> 'pl.Expr'"
    )


def test_plugin_path_points_to_imported_extension() -> None:
    assert PLUGIN_PATH.is_file()
    assert PLUGIN_PATH.name.startswith("_internal")
    assert any(str(PLUGIN_PATH).endswith(suffix) for suffix in EXTENSION_SUFFIXES)


def test_embedding_registers_validated_plugin_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(namespace_module, "compiled_features", lambda: {"embedding"})
    monkeypatch.setattr(
        namespace_module,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs) or pl.lit([0.0]),
    )

    cache_path = tmp_path / "embeddings.duckdb"
    expr = cast(Any, pl.col("text")).text.embedding(
        model=" onnx-community/all-MiniLM-L6-v2-ONNX ",
        cache=cache_path,
        batch_size=16,
    )

    assert isinstance(expr, pl.Expr)
    assert calls[0]["kwargs"] == {
        "embedder_model": "onnx-community/all-MiniLM-L6-v2-ONNX",
        "cache": str(cache_path),
        "batch_size": 16,
    }
    with pytest.raises(ValueError, match="batch_size"):
        cast(Any, pl.col("text")).text.embedding(batch_size=0)


def test_topic_modeling_registers_only_the_supported_fit_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        namespace_module, "compiled_features", lambda: {"topic-modeling"}
    )
    monkeypatch.setattr(
        namespace_module,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs) or pl.lit([0]),
    )

    cast(Any, pl.col("text")).text.topic_modeling(
        segmentation="line", max_tokens=80, seed=7, min_topic_size=2
    )

    assert calls[0]["kwargs"] == {
        "embedder_model": None,
        "cache": None,
        "segmentation_method": "line",
        "max_tokens": 80,
        "seed": 7,
        "min_cluster_size": 2,
        "vectorizer_model": None,
        "lowercase": True,
    }
    with pytest.raises(ValueError, match="automatic.*line.*sentence"):
        cast(Any, pl.col("text")).text.topic_modeling(segmentation="paragraph")
