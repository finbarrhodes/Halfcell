"""Shared test fixtures.

Unit tests here run without network access and, where practical, without the
processed data files — so they stay runnable in CI on a bare checkout.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED = ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def processed_dir():
    return PROCESSED


def requires_processed(name):
    """Skip marker for tests that need a committed processed parquet."""
    path = PROCESSED / name
    return pytest.mark.skipif(
        not path.exists(), reason=f"{name} not present — run scripts/prepare_data.py"
    )
