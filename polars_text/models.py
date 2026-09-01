from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class TokenizerModel:
    model_id: str
    label: str
    languages: tuple[str, ...]


TOKENIZER_MODELS: Final[tuple[TokenizerModel, ...]] = (
    TokenizerModel("native:plain_words_en", "Plain words (English)", ("en",)),
    TokenizerModel("huggingface:bert-base-uncased", "BERT base uncased", ("en",)),
    TokenizerModel("lindera:jieba", "Jieba", ("zh",)),
    TokenizerModel("lindera:cc-cedict", "CC-CEDICT", ("zh",)),
    TokenizerModel("lindera:ja-ipadic", "IPADIC", ("ja",)),
    TokenizerModel("lindera:ja-ipadic-neologd", "IPADIC Neologd", ("ja",)),
    TokenizerModel("lindera:ja-unidic", "UniDic", ("ja",)),
    TokenizerModel("lindera:ko-dic", "ko-dic", ("ko",)),
)


__all__ = ["TOKENIZER_MODELS", "TokenizerModel"]
