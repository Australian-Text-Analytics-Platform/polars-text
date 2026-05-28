"""Lindera integration tests — hit the real GitHub release download path.

These tests download official Lindera dictionary zips and verify the tokenizer
produces morpheme-level JA / KO output. They are gated behind
``@pytest.mark.network`` and an explicit opt-in env var so they no-op in CI /
offline dev environments.

Run locally with::

    POLARS_TEXT_RUN_LINDERA_TESTS=1 uv run pytest -m network \
        polars-text/tests/test_lindera_integration.py
"""

from __future__ import annotations

import os
from typing import Any, cast

import polars as pl
import polars_text
import pytest

_LINDERA_TESTS_ENV = "POLARS_TEXT_RUN_LINDERA_TESTS"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        _LINDERA_TESTS_ENV not in os.environ,
        reason=f"Set {_LINDERA_TESTS_ENV}=1 to run Lindera download tests.",
    ),
]


@pytest.mark.parametrize(
    "model_id, expected_substring",
    [
        # IPADIC: 関西国際空港 segments into 関西 / 国際 / 空港 with
        # standard MeCab+IPADIC morpheme rules. We assert on the
        # individual morpheme rather than the joined surface so the
        # test survives small dict-version drift.
        ("lindera:ja-ipadic", "関西"),
        # UniDic over-segments compared to IPADIC but still recognises
        # 国際 as a morpheme.
        ("lindera:ja-unidic", "国際"),
    ],
)
def test_lindera_ja_tokenize_produces_morphemes(
    model_id: str, expected_substring: str
) -> None:
    df = pl.DataFrame({"text": ["関西国際空港でトートバッグを買った"]})
    # `.text.tokenize` returns List(Struct{token,start,end}); `.item(0)` hands
    # back a polars Series (not a Python list), so extract token strings from
    # its dictionaries.
    result = cast(
        pl.DataFrame,
        df.lazy()
        .select(cast(Any, pl.col("text")).text.tokenize(model=model_id))
        .collect(),
    )
    tokens = [entry["token"] for entry in result.to_series(0).item(0).to_list()]
    assert any(expected_substring == t or expected_substring in t for t in tokens), (
        f"expected {expected_substring!r} morpheme in {model_id} tokens; got {tokens!r}"
    )


def test_lindera_ko_tokenize_produces_morphemes() -> None:
    df = pl.DataFrame({"text": ["한국어 형태소 분석은 흥미롭다"]})
    result = cast(
        pl.DataFrame,
        df.lazy()
        .select(cast(Any, pl.col("text")).text.tokenize(model="lindera:ko-dic"))
        .collect(),
    )
    tokens = [entry["token"] for entry in result.to_series(0).item(0).to_list()]
    # 한국어 = "Korean (language)"; one of the most common standalone
    # nouns. Whatever the exact ko-dic segmentation, this morpheme
    # should be in the output.
    assert any("한국" in t for t in tokens), (
        f"expected a 한국* morpheme in ko-dic tokens; got {tokens!r}"
    )


def test_lindera_offsets_reconstruct_source() -> None:
    """tokenize must emit char offsets that re-slice the
    original text — same invariant as the Jieba offset test in
    test_jieba_chinese.py. Catches the byte-vs-char conversion bug
    that bit us during the topic-modelling CJK fix.
    """
    text = "今日は良い天気"
    df = pl.DataFrame({"text": [text]})
    # `tokenize` returns a List(Struct{token,start,end}).
    # `.to_list()` on the inner Series surfaces a Python list of dicts.
    result = cast(
        pl.DataFrame,
        df.lazy()
        .select(cast(Any, pl.col("text")).text.tokenize(model="lindera:ja-ipadic"))
        .collect(),
    )
    rows = result.to_series(0).item(0).to_list()
    chars = list(text)
    for entry in rows:
        tok = entry["token"]
        start = int(entry["start"])
        end = int(entry["end"])
        # Char-indexed slicing matches the offset convention emitted by
        # tokenize (Jieba does the same).
        extracted = "".join(chars[start:end])
        assert tok == extracted, (
            f"Lindera offset mismatch for {tok!r} ({start}..{end}); "
            f"extracted {extracted!r} from {text!r}"
        )
