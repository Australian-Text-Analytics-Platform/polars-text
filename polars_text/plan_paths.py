"""Inspect and rewrite scan source paths inside a serialized Polars plan.

The backend persists each :class:`~docworkspace.Node` as a ``.plbin`` file
written with ``LazyFrame.serialize(..., format="binary")``. The plan
contains absolute paths baked in at scan time. When a workspace folder is
moved between users (or between machines), those paths become stale.

Two helpers let the backend migrate those paths without touching CWD:

* :func:`list_source_paths` reports every scan source referenced by a
  plan file.
* :func:`replace_source_paths` rewrites paths according to a mapping and
  overwrites the plan file.
"""

from __future__ import annotations

from os import PathLike
from typing import Mapping

from ._internal import list_source_paths as _list_source_paths
from ._internal import replace_source_paths as _replace_source_paths


def list_source_paths(path: str | PathLike[str]) -> list[str]:
    """Return every scan source path referenced by the plan at ``path``.

    Paths are returned in the order they are encountered during a
    depth-first walk of the DSL plan. Duplicates are preserved: if the
    same file is referenced by multiple scan nodes, it appears multiple
    times in the result.
    """

    return _list_source_paths(str(path))


def replace_source_paths(
    path: str | PathLike[str],
    mapper: Mapping[str, str],
) -> int:
    """Rewrite scan source paths in the plan at ``path``.

    ``mapper`` is an exact ``{old_path: new_path}`` substitution table.
    Only paths that appear verbatim as a key are replaced; everything
    else is left untouched. Returns the number of rewritten entries,
    which may exceed ``len(mapper)`` when a mapped path is referenced by
    multiple scan nodes.

    The plan file is only rewritten if at least one substitution
    occurred.
    """

    return _replace_source_paths(str(path), dict(mapper))


__all__ = ["list_source_paths", "replace_source_paths"]
