from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest
from polars_text import _expressions
from polars_text._internal import compiled_features


def test_compiled_features_returns_frozenset() -> None:
    features = compiled_features()

    assert isinstance(features, frozenset)


@pytest.mark.parametrize(
    ("feature", "call"),
    [
        (
            "tokenization",
            lambda: _expressions.tokenize(
                cast(Any, "text"), model="native:plain_words_en"
            ),
        ),
        (
            "tokenization",
            lambda: _expressions.concordance(cast(Any, "text"), "needle"),
        ),
        ("embedding", lambda: _expressions.embedding(cast(Any, "text"))),
        ("topic-modeling", lambda: _expressions.topic_modeling(cast(Any, "text"))),
    ],
)
def test_feature_gated_plugin_wrappers_raise_before_registration(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
    call: Callable[[], object],
) -> None:
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(_expressions, "compiled_features", lambda: frozenset())
    monkeypatch.setattr(
        _expressions,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match=f"requires the '{feature}' feature"):
        call()

    assert calls == []
