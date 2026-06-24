from __future__ import annotations

import ast
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _imports_duckdb(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "duckdb" or alias.name.startswith("duckdb.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and (
            node.module == "duckdb" or (node.module or "").startswith("duckdb.")
        ):
            return True
    return False


class PythonDependencyTests(unittest.TestCase):
    def test_runtime_dependencies_only_include_imported_runtime_packages(self) -> None:
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertEqual(pyproject["project"]["dependencies"], ["polars==1.40.0"])

    def test_python_sources_do_not_import_duckdb_package(self) -> None:
        python_files = [
            *sorted((ROOT / "polars_text").glob("**/*.py")),
            *sorted((ROOT / "tests").glob("test_*.py")),
        ]
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in python_files
            if path.name != Path(__file__).name and _imports_duckdb(path)
        ]

        self.assertEqual(offenders, [])
