"""Shared test fixtures for MorphoCLIP tests."""

import socket
from pathlib import Path

import pytest

from morphoclip.data.metadata import MetadataIndex

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


@pytest.fixture(scope="session")
def metadata_index(metadata_dir: Path) -> MetadataIndex:
    """The fixture platemaps, parsed once.

    Session-scoped because it is read-only: nothing on ``MetadataIndex`` mutates
    it, and ``lookup`` hands back a frozen ``PerturbationInfo``. Do not add a
    mutator without narrowing this scope.
    """
    return MetadataIndex.from_directory(metadata_dir, batch=BATCH)
