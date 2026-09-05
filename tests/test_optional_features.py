from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

import polars_text.namespace as namespace_module
from polars_text.namespace import TextNamespace


@pytest.mark.parametrize(
    ("feature", "call"),
    [
        (
            "tokenization",
            lambda: TextNamespace(cast(Any, "text")).tokenize(
                model="native:plain_words_en"
            ),
        ),
        (
            "tokenization",
            lambda: TextNamespace(cast(Any, "text")).concordance("needle"),
        ),
        (
            "quotation",
            lambda: TextNamespace(cast(Any, "text")).quotation(model_path="/missing"),
        ),
        ("embedding", lambda: TextNamespace(cast(Any, "text")).embedding()),
        ("topic-modeling", lambda: TextNamespace(cast(Any, "text")).topic_modeling()),
    ],
)
def test_feature_gated_plugin_wrappers_raise_before_registration(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
    call: Callable[[], object],
) -> None:
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(namespace_module, "compiled_features", lambda: frozenset())
    monkeypatch.setattr(
        namespace_module,
        "register_plugin_function",
        lambda **kwargs: calls.append(kwargs),
    )

    with pytest.raises(RuntimeError, match=f"requires the '{feature}' feature"):
        call()

    assert calls == []
