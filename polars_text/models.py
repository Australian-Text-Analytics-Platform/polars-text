from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class TokenizerModel:
    model_id: str
    label: str
    languages: tuple[str, ...]


from ._internal import tokenizer_models

TOKENIZER_MODELS: Final[tuple[TokenizerModel, ...]] = tuple(
    TokenizerModel(model_id, label, tuple(languages))
    for model_id, label, languages in tokenizer_models()
)


__all__ = ["TOKENIZER_MODELS", "TokenizerModel"]
