"""Shared test fixtures for MorphoCLIP tests."""

from pathlib import Path

import pytest

# Project root (parent of tests/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Real dataset paths
SAMPLE_PLATE_DIR = PROJECT_ROOT / "data" / "cpjump1" / "BR00116991" / "Images"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
BATCH = "2020_11_04_CPJUMP1"


@pytest.fixture
def sample_plate_dir() -> Path:
    """Path to sample plate images, skips if not present."""
    if not SAMPLE_PLATE_DIR.exists():
        pytest.skip("Sample plate data not available")
    return SAMPLE_PLATE_DIR


@pytest.fixture
def metadata_dir() -> Path:
    """Path to real CPJUMP1 metadata directory, skips if not present."""
    if not METADATA_DIR.exists():
        pytest.skip("Real metadata not available")

    # Quick sanity check that key files exist
    barcode_csv = METADATA_DIR / "platemaps" / BATCH / "barcode_platemap.csv"

    if not barcode_csv.exists():
        pytest.skip(f"barcode_platemap.csv not found at {barcode_csv}")
    return METADATA_DIR
