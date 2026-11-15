"""Test configuration.

Two test populations:

* default (local, pure Python, no torch): selection, accounting, screens,
  textutils, the full orchestration on the synthetic backend, and the frozen
  E3 reference reproduction.
* ``remote`` (GPU + model weights; run on Kaggle / HF Jobs only): the torch
  backend.  ``pytest.ini`` excludes them by default (``-m "not remote"``).

Team protocol: nothing that imports torch runs on the local machine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# make the package importable when pytest is invoked from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "results"


@pytest.fixture
def reference_csv():
    return str(REFERENCE_DIR / "final_panel.csv")


@pytest.fixture
def frozen_claims():
    return __import__("reproduce_frozen").CLAIMS


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "remote: needs GPU + model weights (Kaggle/HF Jobs); never run locally")


def pytest_collection_modifyitems(config, items):
    # fail loudly if a remote test would run in the default (local) session
    if config.getoption("-m", default="not remote") == "not remote":
        for item in items:
            if "remote" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="remote-only (GPU); run with -m remote"))
