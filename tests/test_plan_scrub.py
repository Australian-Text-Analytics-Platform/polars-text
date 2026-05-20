"""Tests for the FFI-plugin scrubber.

Background — polars *does* dead-code-eliminate FFI plugin calls at
collect time when their output column is dropped by a downstream
operation. The first two tests in this module pin that behaviour
empirically: ``lf.drop(col).with_columns(new_expr.alias(col))``
followed by a collect runs only the new expression (the old one's
side effects are suppressed by DCE).

So why do we need this scrubber at all? Because DCE only fires at
collect time — the *serialised* lazy plan still contains the old
expression. A workspace `.plbin` whose owner has been re-tokenised
ten times therefore carries ten dead `tokenize_with_cache_lookup`
expressions, growing unbounded with shares. The scrubber strips them
from the DSL itself, so saved plans stay minimal and load-time mismatch
fixes don't add a new layer on every share.

The scrubber is also the explicit primitive callers reach for when they
want "actually remove this expression from the plan" semantics rather
than relying on optimiser DCE — useful for any future operation where
the column being produced may not be obviously dropped downstream.
"""

from __future__ import annotations

import io
import shutil
from pathlib import Path

import polars as pl
import pytest

import polars_text as pt
from polars_text import scrub_plugin_expressions

JIEBA_TEXT_ROWS = ["我们 都是 good people", "今天 天气 真好"]


def _bucket_dir(cache_root: Path, user_id: str) -> Path:
    return cache_root / user_id / "tokens"


def _tokenize_lf(
    lf: pl.LazyFrame, *, user_id: str, bucket: str, alias: str
) -> pl.LazyFrame:
    return lf.with_columns(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id=user_id,
            bucket_filename=bucket,
            model="jieba",
            lowercase=False,
            remove_punct=True,
        ).alias(alias)
    )


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "ldaca-tokens-cache"
    monkeypatch.setenv("LDACA_TOKENS_CACHE_DIR", str(root))
    return root


def test_polars_dce_at_collect_time(cache_root: Path) -> None:
    """Pin the polars behaviour that makes the *runtime* safety story
    work: when a tokenised column is dropped and re-added under a new
    expression, only the new expression fires at collect. Alice's
    tree must not be touched even though her expression is still
    present in the un-optimised DSL."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})

    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice",
        bucket="bucket-A.parquet",
        alias="__derived__.tokens.text.jieba",
    )
    lf = lf.drop("__derived__.tokens.text.jieba")
    lf = _tokenize_lf(
        lf,
        user_id="bob",
        bucket="bucket-A.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    # Collect via a downstream op that *uses* the tokens column, so DCE
    # cannot just suppress everything.
    result = lf.select(
        pl.col("__derived__.tokens.text.jieba").list.len().alias("n")
    ).collect()
    assert result["n"].to_list() == [5, 4]

    alice_dir = _bucket_dir(cache_root, "alice")
    bob_dir = _bucket_dir(cache_root, "bob")
    assert not alice_dir.exists() or not any(alice_dir.iterdir()), (
        "alice's tree must not be touched after Bob's drop+re-add"
    )
    assert bob_dir.exists() and any(bob_dir.iterdir()), (
        "bob's tree should receive the cache parquet"
    )


def test_saved_plan_still_carries_dropped_expression(cache_root: Path) -> None:
    """DCE is a collect-time optimisation, NOT a plan rewrite. The
    serialised plan keeps the dropped expression intact — this is the
    bloat the scrubber exists to clear."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})

    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice-the-original",
        bucket="bucket-B.parquet",
        alias="__derived__.tokens.text.jieba",
    )
    lf = lf.drop("__derived__.tokens.text.jieba")
    lf = _tokenize_lf(
        lf,
        user_id="bob-the-importer",
        bucket="bucket-B.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    saved = lf.serialize(format="binary")
    # Both user_ids are still in the plan bytes.
    assert b"alice-the-original" in saved
    assert b"bob-the-importer" in saved


def test_scrub_removes_aliased_plugin_expression(cache_root: Path) -> None:
    """Happy path: alias name + plugin symbol both match → expression
    is filtered out of HStack, and the saved plan no longer carries
    the targeted user_id."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice-the-original",
        bucket="bucket-C.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    scrubbed, removed = scrub_plugin_expressions(
        lf,
        aliases=["__derived__.tokens.text.jieba"],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 1
    assert b"alice-the-original" not in scrubbed.serialize(format="binary")


def test_scrub_then_readd_keeps_plan_minimal(cache_root: Path) -> None:
    """The intended caller pattern: scrub the old expression, then add
    a new one. The resulting plan must contain ONLY the new user's
    identity, not the original author's."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice-the-original",
        bucket="bucket-D.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    scrubbed, removed = scrub_plugin_expressions(
        lf,
        aliases=["__derived__.tokens.text.jieba"],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 1
    retokenised = _tokenize_lf(
        scrubbed,
        user_id="bob-the-importer",
        bucket="bucket-D.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    plan_bytes = retokenised.serialize(format="binary")
    assert b"alice-the-original" not in plan_bytes
    assert b"bob-the-importer" in plan_bytes

    # And the collect path still works end-to-end on bob's tree.
    for u in ("alice-the-original", "bob-the-importer"):
        d = _bucket_dir(cache_root, u)
        if d.exists():
            shutil.rmtree(d)

    result = retokenised.select(
        pl.col("__derived__.tokens.text.jieba").list.len().alias("n")
    ).collect()
    assert result["n"].to_list() == [5, 4]
    assert _bucket_dir(cache_root, "bob-the-importer").exists()
    assert not _bucket_dir(cache_root, "alice-the-original").exists()


def test_scrub_returns_unchanged_lf_when_no_match(cache_root: Path) -> None:
    """No alias matches → returns the input LazyFrame untouched and
    ``removed == 0``. Callers can short-circuit without a
    deserialize round-trip."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice",
        bucket="bucket-E.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    scrubbed, removed = scrub_plugin_expressions(
        lf,
        aliases=["__derived__.tokens.text.something-else"],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 0
    assert scrubbed is lf


def test_scrub_only_removes_when_plugin_symbol_matches(cache_root: Path) -> None:
    """Safety guard: an alias that matches the target name but whose
    inner expression is a *different* plugin (or not a plugin at all)
    is left alone. Without this, the scrubber could accidentally tear
    out user-built columns that share a derived-style name."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = df.lazy().with_columns(
        pl.col("text").str.to_uppercase().alias("__derived__.tokens.text.jieba")
    )

    scrubbed, removed = scrub_plugin_expressions(
        lf,
        aliases=["__derived__.tokens.text.jieba"],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 0
    assert scrubbed is lf
    result = lf.collect()
    assert result["__derived__.tokens.text.jieba"].to_list() == [
        s.upper() for s in JIEBA_TEXT_ROWS
    ]


def test_scrub_handles_multiple_targeted_aliases(cache_root: Path) -> None:
    """Two tokenised columns in one plan → both removed in one pass."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice",
        bucket="bucket-F.parquet",
        alias="__derived__.tokens.text.jieba",
    )
    lf = _tokenize_lf(
        lf,
        user_id="alice",
        bucket="bucket-G.parquet",
        alias="__derived__.tokens.text.bert",
    )

    scrubbed, removed = scrub_plugin_expressions(
        lf,
        aliases=[
            "__derived__.tokens.text.jieba",
            "__derived__.tokens.text.bert",
        ],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 2
    plan_bytes = scrubbed.serialize(format="binary")
    assert b"bucket-F.parquet" not in plan_bytes
    assert b"bucket-G.parquet" not in plan_bytes


def test_scrub_partial_match_only_removes_named_aliases(cache_root: Path) -> None:
    """Only the listed aliases are removed; siblings stay intact."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice",
        bucket="bucket-H.parquet",
        alias="__derived__.tokens.text.jieba",
    )
    lf = _tokenize_lf(
        lf,
        user_id="alice",
        bucket="bucket-I.parquet",
        alias="__derived__.tokens.text.bert",
    )

    scrubbed, removed = scrub_plugin_expressions(
        lf,
        aliases=["__derived__.tokens.text.jieba"],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 1
    plan_bytes = scrubbed.serialize(format="binary")
    assert b"bucket-H.parquet" not in plan_bytes
    assert b"bucket-I.parquet" in plan_bytes


def test_scrub_empty_alias_list_is_noop(cache_root: Path) -> None:
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice",
        bucket="bucket-J.parquet",
        alias="__derived__.tokens.text.jieba",
    )
    scrubbed, removed = scrub_plugin_expressions(
        lf, aliases=[], symbol="tokenize_with_cache_lookup"
    )
    assert removed == 0
    assert scrubbed is lf


def test_scrub_survives_serialize_roundtrip(cache_root: Path) -> None:
    """Production flow: a plan is serialized to .plbin, the bytes are
    later read back, scrubbed, and re-serialized. The post-scrub bytes
    must no longer carry the targeted expression."""
    df = pl.DataFrame({"text": JIEBA_TEXT_ROWS})
    lf = _tokenize_lf(
        df.lazy(),
        user_id="alice-the-original",
        bucket="bucket-K.parquet",
        alias="__derived__.tokens.text.jieba",
    )

    plbin_bytes = lf.serialize(format="binary")
    reloaded = pl.LazyFrame.deserialize(io.BytesIO(plbin_bytes), format="binary")

    scrubbed, removed = scrub_plugin_expressions(
        reloaded,
        aliases=["__derived__.tokens.text.jieba"],
        symbol="tokenize_with_cache_lookup",
    )
    assert removed == 1
    scrubbed_bytes = scrubbed.serialize(format="binary")
    assert b"alice-the-original" not in scrubbed_bytes
