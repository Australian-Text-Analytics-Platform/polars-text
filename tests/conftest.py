import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--require-models",
        action="store_true",
        help="Require provisioned quotation model tests to run",
    )


def pytest_configure(config):
    if config.getoption("--require-models"):
        path = os.environ.get("WORDFLOW_TEST_UDPIPE_MODEL")
        if not path or not Path(path).is_file():
            raise pytest.UsageError(
                "--require-models needs WORDFLOW_TEST_UDPIPE_MODEL pointing to the provisioned model"
            )
