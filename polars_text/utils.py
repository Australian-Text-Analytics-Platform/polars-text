from __future__ import annotations

from pathlib import Path

from . import _internal

PLUGIN_PATH = Path(_internal.__file__).resolve()
