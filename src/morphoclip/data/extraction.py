"""Plate selection and output bookkeeping for `morphoclip features extract`."""

from dataclasses import dataclass
from pathlib import Path

from morphoclip.data.feature_extractor import verify_plate_features
from morphoclip.data.perturbation import extract_plate_barcode


@dataclass(frozen=True, slots=True)
class PlateOutputState:
    """What a plate's feature directory held before extraction.

    Attributes:
        extracted: Feature files found.
        expected: Feature files the plate's images call for.
        complete: Every expected feature file is present, so extraction can skip.
        cleared_features: Feature files removed from a partial run.
        cleared_tensors: Tensor files removed from a partial run.
    """

    extracted: int
    expected: int
    complete: bool
    cleared_features: int = 0
    cleared_tensors: int = 0


def select_plates(plates: list[str], plate: str | None) -> list[str]:
    """Restrict *plates* to *plate*, matched by full name or barcode.

    Args:
        plates: Plate names from the dataset config.
        plate: Requested plate, or ``None`` for all of them.

    Returns:
        The matching plates, or ``[plate]`` when none match so an unlisted
        plate can still be extracted.
    """
    if not plate:
        return plates
    matches = [p for p in plates if extract_plate_barcode(p) == plate or p == plate]
    return matches or [plate]


def resolve_image_dir(compressed_root: Path, batch: str, plate_name: str) -> Path | None:
    """Find a plate's ``Images/`` directory, with or without the batch level.

    Args:
        compressed_root: Root of the compressed images.
        batch: Batch directory name.
        plate_name: Plate directory name.

    Returns:
        ``<root>/<batch>/<plate>/Images`` if it exists, else
        ``<root>/<plate>/Images`` if that exists, else ``None``.
    """
    for image_dir in (
        compressed_root / batch / plate_name / "Images",
        compressed_root / plate_name / "Images",
    ):
        if image_dir.exists():
            return image_dir
    return None


def clear_pt_files(directory: Path) -> int:
    """Remove saved ``.pt`` files from *directory*, returning how many went."""
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob("*.pt"):
        path.unlink()
        removed += 1
    return removed


def prepare_plate_output(
    image_dir: Path,
    feature_dir: Path,
    tensor_dir: Path | None,
) -> PlateOutputState:
    """Check a plate's existing output and clear it if a previous run stopped partway.

    A partial directory is wiped rather than topped up, so a plate never mixes
    files from two runs.

    Args:
        image_dir: The plate's ``Images/`` directory.
        feature_dir: Where the plate's feature files go.
        tensor_dir: Where its resized tensors go, or ``None`` when they are
            not saved (and so must not be cleared).

    Returns:
        The state found, with the counts of anything cleared.
    """
    extracted, expected, missing = verify_plate_features(feature_dir, image_dir)
    if extracted == expected and not missing:
        return PlateOutputState(extracted=extracted, expected=expected, complete=True)
    if extracted == 0:
        return PlateOutputState(extracted=extracted, expected=expected, complete=False)
    return PlateOutputState(
        extracted=extracted,
        expected=expected,
        complete=False,
        cleared_features=clear_pt_files(feature_dir),
        cleared_tensors=clear_pt_files(tensor_dir) if tensor_dir is not None else 0,
    )
