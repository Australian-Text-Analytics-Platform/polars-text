"""Tests for plan source path introspection and rewriting."""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
from polars_text import list_source_paths, replace_source_paths


@pytest.fixture
def parquet_dataset(tmp_path: Path) -> Path:
    df = pl.DataFrame({"id": [1, 2, 3], "text": ["a", "b", "c"]})
    target = tmp_path / "orig" / "data.parquet"
    target.parent.mkdir(parents=True)
    df.write_parquet(target)
    return target


def _serialize_lazy(lf: pl.LazyFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    lf.serialize(dest, format="binary")


def test_list_source_paths_simple(parquet_dataset: Path, tmp_path: Path) -> None:
    plbin = tmp_path / "plans" / "node.plbin"
    lf = pl.scan_parquet(parquet_dataset)
    _serialize_lazy(lf, plbin)

    paths = list_source_paths(plbin)
    assert len(paths) == 1
    assert Path(paths[0]).resolve() == parquet_dataset.resolve()


def test_list_source_paths_through_transforms(
    parquet_dataset: Path, tmp_path: Path
) -> None:
    """The scan source must still be reported when transforms wrap the scan."""

    plbin = tmp_path / "plans" / "node.plbin"
    lf = (
        pl.scan_parquet(parquet_dataset).filter(pl.col("id") > 1).select(["id", "text"])
    )
    _serialize_lazy(lf, plbin)

    paths = list_source_paths(plbin)
    assert len(paths) == 1
    assert Path(paths[0]).resolve() == parquet_dataset.resolve()


def test_replace_source_paths_round_trip(parquet_dataset: Path, tmp_path: Path) -> None:
    plbin = tmp_path / "workspace_a" / "data" / "node.plbin"
    lf = pl.scan_parquet(parquet_dataset)
    _serialize_lazy(lf, plbin)

    # Move the actual parquet to a different directory, as if the whole
    # workspace folder had been moved between users.
    new_dir = tmp_path / "workspace_b" / "data"
    new_dir.mkdir(parents=True)
    new_parquet = new_dir / parquet_dataset.name
    shutil.move(str(parquet_dataset), str(new_parquet))

    original_paths = list_source_paths(plbin)
    assert len(original_paths) == 1
    old_path = original_paths[0]

    changed = replace_source_paths(plbin, {old_path: str(new_parquet)})
    assert changed == 1

    rewritten_paths = list_source_paths(plbin)
    assert len(rewritten_paths) == 1
    assert Path(rewritten_paths[0]).resolve() == new_parquet.resolve()

    # The rewritten plan must still be deserializable and collectable.
    reloaded = pl.LazyFrame.deserialize(plbin, format="binary")
    result = reloaded.collect()
    assert result.shape == (3, 2)


def test_replace_source_paths_noop_when_no_match(
    parquet_dataset: Path, tmp_path: Path
) -> None:
    plbin = tmp_path / "node.plbin"
    lf = pl.scan_parquet(parquet_dataset)
    _serialize_lazy(lf, plbin)

    changed = replace_source_paths(plbin, {"/nonexistent/foo.parquet": "/bar.parquet"})
    assert changed == 0

    paths = list_source_paths(plbin)
    assert len(paths) == 1
    assert Path(paths[0]).resolve() == parquet_dataset.resolve()


def test_replace_source_paths_multiple_scans(tmp_path: Path) -> None:
    """A plan that joins two scans should rewrite both when mapped."""

    data_a = tmp_path / "a.parquet"
    data_b = tmp_path / "b.parquet"
    pl.DataFrame({"id": [1, 2], "x": [10, 20]}).write_parquet(data_a)
    pl.DataFrame({"id": [1, 2], "y": [100, 200]}).write_parquet(data_b)

    plbin = tmp_path / "join.plbin"
    lf = pl.scan_parquet(data_a).join(pl.scan_parquet(data_b), on="id", how="inner")
    _serialize_lazy(lf, plbin)

    paths = list_source_paths(plbin)
    assert len(paths) == 2

    new_dir = tmp_path / "moved"
    new_dir.mkdir()
    moved_a = new_dir / "a.parquet"
    moved_b = new_dir / "b.parquet"
    shutil.move(str(data_a), str(moved_a))
    shutil.move(str(data_b), str(moved_b))

    changed = replace_source_paths(
        plbin,
        {paths[0]: str(moved_a), paths[1]: str(moved_b)},
    )
    assert changed == 2

    reloaded = pl.LazyFrame.deserialize(plbin, format="binary")
    result = reloaded.collect()
    assert result.shape == (2, 3)


def test_parquet_source_path_replace_with_data_validation(tmp_path: Path) -> None:
    """Create a toy parquet, rewrite the scan path, and verify cell-level data."""

    original_dir = tmp_path / "project_v1"
    original_dir.mkdir()
    parquet_path = original_dir / "corpus.parquet"
    expected = pl.DataFrame(
        {
            "doc_id": ["d1", "d2", "d3", "d4"],
            "word_count": [120, 340, 56, 890],
            "lang": ["en", "fr", "en", "de"],
        }
    )
    expected.write_parquet(parquet_path)

    plbin = tmp_path / "plans" / "analysis.plbin"
    lf = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("word_count") > 100)
        .select(["doc_id", "word_count"])
    )
    _serialize_lazy(lf, plbin)

    # Simulate moving the workspace to a new location.
    relocated_dir = tmp_path / "project_v2"
    shutil.copytree(str(original_dir), str(relocated_dir))
    shutil.rmtree(str(original_dir))

    relocated_parquet = relocated_dir / "corpus.parquet"
    assert relocated_parquet.exists()
    assert not parquet_path.exists()

    old_paths = list_source_paths(plbin)
    changed = replace_source_paths(plbin, {old_paths[0]: str(relocated_parquet)})
    assert changed == 1

    result = pl.LazyFrame.deserialize(plbin, format="binary").collect()
    assert result.shape == (3, 2)
    assert result["doc_id"].to_list() == ["d1", "d2", "d4"]
    assert result["word_count"].to_list() == [120, 340, 890]
