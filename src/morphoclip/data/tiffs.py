"""Counting and deleting the raw TIFFs a downloaded plate arrives as.

Both extensions are checked because the gallery uses ``.tiff`` on some plates
and ``.tif`` on others.
"""

from pathlib import Path

TIFF_SUFFIXES = ("*.tif", "*.tiff")


def tiff_paths(image_dir: Path) -> list[Path]:
    """Sorted TIFF paths directly under *image_dir*."""
    return sorted(path for suffix in TIFF_SUFFIXES for path in image_dir.glob(suffix))


def count_tiffs(image_dir: Path) -> int:
    """Number of TIFFs in *image_dir*, or 0 if it does not exist."""
    if not image_dir.exists():
        return 0
    return len(tiff_paths(image_dir))


def delete_tiffs(image_dir: Path, *, dry_run: bool = False) -> int:
    """Delete the plate's TIFFs once features are extracted.

    Returns:
        How many files were deleted, or would have been under *dry_run*.
    """
    paths = tiff_paths(image_dir)
    if not dry_run:
        for path in paths:
            path.unlink(missing_ok=True)
    return len(paths)
