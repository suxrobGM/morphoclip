"""Tests for morphoclip.data.tiffs, the destructive half of the extraction pipeline."""

from pathlib import Path

import pytest

from morphoclip.data.tiffs import count_tiffs, delete_tiffs, tiff_paths

# Names interleave the two extensions so a concatenation of the two globs
# would come out in a different order than a sort of both together.
TIFFS = ("a.tif", "c.tif", "b.tiff", "d.tiff")
KEEP = "Index.idx.xml"


@pytest.fixture
def image_dir(tmp_path: Path) -> Path:
    """A plate ``Images/`` directory holding two .tif, two .tiff and one .xml."""
    directory = tmp_path / "Images"
    directory.mkdir()
    for name in (*TIFFS, KEEP):
        (directory / name).write_bytes(b"not really a tiff")
    return directory


def test_count_covers_both_tif_and_tiff_and_ignores_other_files(image_dir: Path) -> None:
    assert count_tiffs(image_dir) == len(TIFFS)


def test_count_is_zero_for_a_directory_that_was_never_downloaded(tmp_path: Path) -> None:
    assert count_tiffs(tmp_path / "BR00116991" / "Images") == 0


def test_paths_are_sorted_across_both_extensions(image_dir: Path) -> None:
    assert [path.name for path in tiff_paths(image_dir)] == sorted(TIFFS)


def test_dry_run_reports_the_count_without_deleting(image_dir: Path) -> None:
    assert delete_tiffs(image_dir, dry_run=True) == len(TIFFS)
    assert {path.name for path in image_dir.iterdir()} == {*TIFFS, KEEP}


def test_delete_removes_every_tiff_and_leaves_the_rest(image_dir: Path) -> None:
    assert delete_tiffs(image_dir) == len(TIFFS)
    assert {path.name for path in image_dir.iterdir()} == {KEEP}


def test_deleting_an_already_cleaned_directory_reports_nothing(image_dir: Path) -> None:
    delete_tiffs(image_dir)
    assert delete_tiffs(image_dir) == 0
