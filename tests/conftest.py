"""Shared test fixtures for MorphoCLIP tests."""

import socket
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_METADATA_DIR = Path(__file__).parent / "fixtures" / "cpjump1" / "metadata"
DOWNLOADED_METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
BATCH = "2020_11_04_CPJUMP1"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly on an unpatched network call instead of hanging CI."""

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("Tests must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", blocked)


@pytest.fixture(scope="session")
def metadata_dir() -> Path:
    """Committed CPJUMP1 platemap tree. See tests/fixtures/README.md."""
    return FIXTURE_METADATA_DIR
