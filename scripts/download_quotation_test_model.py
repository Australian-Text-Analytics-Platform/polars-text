"""Delegate model/source provisioning to the owning native crate."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2] / "ldaca-rs" / "scripts" / "download_quotation_test_model.py"), run_name="__main__")
