"""Phase 5 Lindera integration tests — hit the real download path.

These tests download a Lindera dict from the configured HF repo
(``LDACA_LINDERA_DICT_REPO``, default ``ldaca/lindera-dicts``) and
verify the tokenizer produces morpheme-level JA / KO output. They are
gated behind ``@pytest.mark.network`` AND a ``skipif`` checking for
``LDACA_LINDERA_DICT_REPO`` so they no-op in CI / offline dev environments
until the dict-hosting decision is finalized.

Run locally with::

    LDACA_LINDERA_DICT_REPO=mily/lindera-dicts uv run pytest -m network \
        polars-text/tests/test_lindera_integration.py
"""

from __future__ import annotations

import os

import polars as pl
import polars_text as pt
import pytest

_DICT_REPO_ENV = "LDACA_LINDERA_DICT_REPO"

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        _DICT_REPO_ENV not in os.environ,
        reason=(
            f"Set {_DICT_REPO_ENV} to a reachable HF dataset hosting the "
            "Lindera dict tarballs to enable these tests."
        ),
    ),
]


@pytest.mark.parametrize(
    "model_id, expected_substring",
    [
        # IPADIC: 関西国際空港 segments into 関西 / 国際 / 空港 with
        # standard MeCab+IPADIC morpheme rules. We assert on the
        # individual morpheme rather than the joined surface so the
        # test survives small dict-version drift.
        ("lindera-ja-ipadic", "関西"),
        # UniDic over-segments compared to IPADIC but still recognises
        # 国際 as a morpheme.
        ("lindera-ja-unidic", "国際"),
    ],
)
def test_lindera_ja_tokenize_produces_morphemes(
    model_id: str, expected_substring: str
) -> None:
    df = pl.DataFrame({"text": ["関西国際空港でトートバッグを買った"]})
    tokens = (
        df.lazy()
        .select(pt.tokenize(pl.col("text"), model=model_id))
        .collect()
        .to_series(0)
        .item(0)
    )
    assert isinstance(tokens, list)
    assert any(
        expected_substring == t or expected_substring in t for t in tokens
    ), (
        f"expected {expected_substring!r} morpheme in {model_id} tokens; "
        f"got {tokens!r}"
    )


def test_lindera_ko_tokenize_produces_morphemes() -> None:
    df = pl.DataFrame({"text": ["한국어 형태소 분석은 흥미롭다"]})
    tokens = (
        df.lazy()
        .select(pt.tokenize(pl.col("text"), model="lindera-ko-dic"))
        .collect()
        .to_series(0)
        .item(0)
    )
    assert isinstance(tokens, list)
    # 한국어 = "Korean (language)"; one of the most common standalone
    # nouns. Whatever the exact ko-dic segmentation, this morpheme
    # should be in the output.
    assert any("한국" in t for t in tokens), (
        f"expected a 한국* morpheme in ko-dic tokens; got {tokens!r}"
    )


def test_lindera_offsets_reconstruct_source() -> None:
    """tokenize_with_offsets must emit char offsets that re-slice the
    original text — same invariant as the Jieba offset test in
    test_jieba_chinese.py. Catches the byte-vs-char conversion bug
    that bit us during the topic-modelling CJK fix.
    """
    text = "今日は良い天気"
    df = pl.DataFrame({"text": [text]})
    rows = (
        df.lazy()
        .select(pt.tokenize_with_offsets(pl.col("text"), model="lindera-ja-ipadic"))
        .collect()
        .to_series(0)
        .item(0)
    )
    assert isinstance(rows, list)
    for entry in rows:
        # tokenize_with_offsets emits a list of structs; the
        # to_python conversion above hands us dicts.
        tok = entry["token"]
        start = int(entry["start"])
        end = int(entry["end"])
        extracted = text[start:end] if False else "".join(
            list(text)[start:end]
        )
        # Use char-indexed slicing to match the offset convention.
        chars = list(text)
        extracted = "".join(chars[start:end])
        assert tok == extracted, (
            f"Lindera offset mismatch for {tok!r} ({start}..{end}); "
            f"extracted {extracted!r} from {text!r}"
        )
