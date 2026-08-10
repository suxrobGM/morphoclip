"""Drift alarm for the committed metadata fixture.

Marked ``realdata`` and excluded from the default run, since it needs the
downloaded ``data/metadata/`` tree. Run it with ``uv run poe test-realdata``
after ``morphoclip data fetch`` to confirm the fixture still matches upstream.
"""

from pathlib import Path

import pytest

from tests.conftest import DOWNLOADED_METADATA_DIR, FIXTURE_METADATA_DIR

pytestmark = pytest.mark.realdata


def _relative_files(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_fixture_matches_downloaded_metadata() -> None:
    if not DOWNLOADED_METADATA_DIR.exists():
        pytest.skip(f"No downloaded metadata at {DOWNLOADED_METADATA_DIR}")

    fixture = _relative_files(FIXTURE_METADATA_DIR)
    downloaded = _relative_files(DOWNLOADED_METADATA_DIR)

    assert set(fixture) == set(downloaded)
    drifted = sorted(str(name) for name, blob in fixture.items() if downloaded[name] != blob)
    assert not drifted, f"Fixture is stale, re-copy these into tests/fixtures/: {drifted}"
