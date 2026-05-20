"""Phase 1 — `tokenize_with_cache_lookup` lazy on-demand tokenisation.

The expression takes (source_text, precomputed_hash) and produces the
same `List<Struct<token, start, end>>` schema as `tokenize_with_offsets`,
but with a side-effecting per-row cache:

* cache HIT → return the cached row from disk, skip the tokeniser
* cache MISS → tokenise, return the result, append `(hash, tokens)`
  to a fresh `<bucket>__delta__<uuid>.parquet` under an advisory flock

These tests cover the happy paths + the load → hit-rate behaviour. The
Rust-side unit tests in `src/tokens_cache_io.rs` cover the lower-level
file-layout invariants; this suite exercises the integrated expression
through the registered plugin path so we catch FFI / kwargs-serde /
schema-mismatch regressions.

See: backend/docs/developer-guide/lazy-tokenisation-refactor.md §3-§6.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

import polars_text as pt


CONTENT_HASH_COLUMN = "__ldaca_content_hash__"


@pytest.fixture
def cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Per-test cache base. The Rust expression resolves
    `LDACA_TOKENS_CACHE_DIR + user_id + tokens/`, so every test gets a
    clean directory tree and tests don't see each other's writes."""
    monkeypatch.setenv("LDACA_TOKENS_CACHE_DIR", str(tmp_path))
    return tmp_path


def _list_bucket_files(cache_root: Path, user_id: str, bucket: str) -> list[Path]:
    """List the legacy + delta parquet files for one bucket on disk —
    used by tests to assert disk-side state without going through the
    Rust API directly."""
    d = cache_root / user_id / "tokens"
    if not d.exists():
        return []
    stem = bucket.removesuffix(".parquet")
    files: list[Path] = []
    legacy = d / f"{stem}.parquet"
    if legacy.exists():
        files.append(legacy)
    files.extend(d.glob(f"{stem}__delta__*.parquet"))
    return sorted(files)


def test_first_collect_is_cache_miss_and_writes_delta(cache_root: Path) -> None:
    df = pl.DataFrame({"text": ["hello world", "foo bar baz"]})
    out = df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="alice",
            bucket_filename="testmodel__abc123.parquet",
        )
    )
    # Schema is the contract — `List<Struct<token, start, end>>`
    dtype = out.schema["text"]
    assert isinstance(dtype, pl.List)
    inner = dtype.inner
    assert isinstance(inner, pl.Struct)
    assert {f.name for f in inner.fields} == {"token", "start", "end"}

    # Both rows should have produced tokens
    rows = out["text"].to_list()
    assert len(rows) == 2
    assert all(len(r) > 0 for r in rows), f"expected tokens for both rows, got {rows}"

    # A delta parquet was written under the cache root
    files = _list_bucket_files(cache_root, "alice", "testmodel__abc123.parquet")
    assert len(files) == 1, f"expected 1 delta file, got {files}"
    cache_df = pl.read_parquet(files[0])
    assert set(cache_df.columns) == {CONTENT_HASH_COLUMN, "tokens"}
    assert cache_df.height == 2


def test_second_collect_is_full_cache_hit(cache_root: Path) -> None:
    df = pl.DataFrame({"text": ["one two three", "alpha beta"]})
    expr = pt.tokenize_with_cache_lookup(
        pl.col("text"),
        user_id="bob",
        bucket_filename="testmodel__cafe.parquet",
    )
    out1 = df.select(expr).rename({"text": "tokens"})
    # First collect populates the cache with one delta file
    files1 = _list_bucket_files(cache_root, "bob", "testmodel__cafe.parquet")
    assert len(files1) == 1

    # Second collect with the SAME rows should be all hits → no new
    # delta file written, and the result must match the first.
    out2 = df.select(expr).rename({"text": "tokens"})
    files2 = _list_bucket_files(cache_root, "bob", "testmodel__cafe.parquet")
    assert files2 == files1, "no new delta file expected on full-hit collect"
    assert out1.equals(out2), "second collect must reproduce the first"


def test_mixed_hit_and_miss_appends_only_misses(cache_root: Path) -> None:
    # Seed the cache with one row
    seed_df = pl.DataFrame({"text": ["seeded row"]})
    expr = pt.tokenize_with_cache_lookup(
        pl.col("text"),
        user_id="carol",
        bucket_filename="testmodel__dead.parquet",
    )
    seed_df.select(expr)
    seeded_files = _list_bucket_files(cache_root, "carol", "testmodel__dead.parquet")
    assert len(seeded_files) == 1
    seeded_rows = pl.read_parquet(seeded_files[0]).height
    assert seeded_rows == 1

    # Now collect with one cached row + two fresh rows. Only the two
    # new rows should be appended to a new delta file.
    mixed_df = pl.DataFrame({"text": ["seeded row", "fresh one", "fresh two"]})
    mixed_df.select(expr)
    after_files = _list_bucket_files(cache_root, "carol", "testmodel__dead.parquet")
    assert len(after_files) == 2, (
        f"expected one seed delta + one mixed delta, got {after_files}"
    )
    # The newer file holds exactly the two misses
    new_file = next(p for p in after_files if p not in seeded_files)
    new_rows = pl.read_parquet(new_file)
    assert new_rows.height == 2


def test_missing_env_with_require_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure the env is unset for this test specifically
    monkeypatch.delenv("LDACA_TOKENS_CACHE_DIR", raising=False)
    df = pl.DataFrame({"text": ["anything"]})
    with pytest.raises(pl.exceptions.ComputeError, match="LDACA_TOKENS_CACHE_DIR"):
        df.select(
            pt.tokenize_with_cache_lookup(
                pl.col("text"),
                user_id="someone",
                bucket_filename="b__0.parquet",
                require_env_cache_dir=True,
            )
        )


def test_empty_input_short_circuits(cache_root: Path) -> None:
    # An empty DataFrame should produce an empty result and write no delta.
    df = pl.DataFrame({"text": pl.Series("text", [], dtype=pl.String)})
    out = df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="dan",
            bucket_filename="testmodel__empty.parquet",
        )
    )
    assert out.height == 0
    files = _list_bucket_files(cache_root, "dan", "testmodel__empty.parquet")
    assert files == [], "empty input should write no delta files"


def test_null_text_emits_empty_list_and_skips_cache(cache_root: Path) -> None:
    # A null source row should produce an empty token list and must not
    # write a row to the cache (no hash → no key).
    df = pl.DataFrame({"text": ["something", None, "another"]})
    out = df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="erin",
            bucket_filename="testmodel__nulls.parquet",
        )
    )
    rows = out["text"].to_list()
    assert len(rows) == 3
    # Middle row (null) must surface as empty token list
    assert rows[1] == [] or rows[1] is None, f"expected empty/None for null row, got {rows[1]}"

    files = _list_bucket_files(cache_root, "erin", "testmodel__nulls.parquet")
    assert len(files) == 1
    cached = pl.read_parquet(files[0])
    # Only the two non-null rows should be cached
    assert cached.height == 2


def test_different_bucket_files_are_isolated(cache_root: Path) -> None:
    df = pl.DataFrame({"text": ["shared text"]})
    df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="frank",
            bucket_filename="modelA__0001.parquet",
        )
    )
    df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="frank",
            bucket_filename="modelB__0002.parquet",
        )
    )
    a = _list_bucket_files(cache_root, "frank", "modelA__0001.parquet")
    b = _list_bucket_files(cache_root, "frank", "modelB__0002.parquet")
    assert len(a) == 1 and len(b) == 1
    assert a[0].name != b[0].name


def test_per_user_subdir_isolation(cache_root: Path) -> None:
    df = pl.DataFrame({"text": ["xyz"]})
    df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="user_one",
            bucket_filename="m__h.parquet",
        )
    )
    df.select(
        pt.tokenize_with_cache_lookup(
            pl.col("text"),
            user_id="user_two",
            bucket_filename="m__h.parquet",
        )
    )
    f1 = _list_bucket_files(cache_root, "user_one", "m__h.parquet")
    f2 = _list_bucket_files(cache_root, "user_two", "m__h.parquet")
    assert len(f1) == 1 and len(f2) == 1
    assert f1[0].parent != f2[0].parent
