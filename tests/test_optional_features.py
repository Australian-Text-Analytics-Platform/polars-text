from __future__ import annotations

from typing import Any, Callable

import pytest

from polars_text import functions
from polars_text._internal import compiled_features


def test_compiled_features_returns_frozenset() -> None:
    features = compiled_features()

    assert isinstance(features, frozenset)


@pytest.mark.parametrize(
    ("feature", "call"),
    [
        (
            "tokenization",
            lambda: functions.tokenize("text", model="native:plain_words_en"),
        ),
        ("tokenization", lambda: functions.concordance("text", "needle")),
        ("embedding", lambda: functions.embedding("text")),
        ("topic-modeling", lambda: functions.topic_modeling("text")),
    ],
)
def test_feature_gated_plugin_wrappers_raise_before_registration(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
    call: Callable[[], object],
) -> None:
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(functions, "compiled_features", lambda: frozenset())
    monkeypatch.setattr(
        functions,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match=f"requires the '{feature}' feature"):
        call()

    assert calls == []
