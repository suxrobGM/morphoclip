"""Builder for synthetic CPJUMP1 plate image directories."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from morphoclip.data.image_loader import FLUORESCENCE_CHANNELS

DEFAULT_WELLS: tuple[str, ...] = ("A01", "A03")
DEFAULT_SIZE = 32


def well_to_row_col(well: str) -> tuple[int, int]:
    """Turn ``"A01"`` into ``(1, 1)``."""
    return ord(well[0].upper()) - 64, int(well[1:])


def site_filename(row: int, col: int, field: int, channel: int) -> str:
    """CPJUMP1 microscopy filename for one channel of one site."""
    return f"r{row:02d}c{col:02d}f{field:02d}p01-ch{channel}sk1fk1fl1.tiff"


def _pixels(row: int, col: int, field: int, channel: int, size: int) -> np.ndarray:
    """Deterministic uint16 pixels, distinct per (site, channel)."""
    seed = (row * 1_000_000) + (col * 10_000) + (field * 100) + channel
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65536, size=(size, size), dtype=np.uint16)


def make_plate_images(
    image_dir: Path,
    *,
    wells: Sequence[str] = DEFAULT_WELLS,
    sites: int = 2,
    channels: Sequence[int] = FLUORESCENCE_CHANNELS,
    size: int = DEFAULT_SIZE,
    extra_files: bool = True,
    incomplete_well: str | None = None,
) -> Path:
    """Write a plate ``Images/`` directory of synthetic uint16 TIFFs.

    Args:
        image_dir: Directory to create and fill.
        wells: Well labels like ``"A01"``.
        sites: Fields per well, numbered from 1.
        channels: Channel numbers to write for every site.
        size: Image edge length in pixels.
        extra_files: Also write brightfield ch6-ch8 and an ``Index.idx.xml``,
            so the channel filter and the non-image filter get exercised.
        incomplete_well: Well whose first site is missing its last channel, so
            the "only complete sites" filter gets exercised. Must not be in
            *wells* if you want the well itself excluded entirely.

    Returns:
        ``image_dir``.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    targets = list(wells)
    if incomplete_well is not None and incomplete_well not in targets:
        targets.append(incomplete_well)

    for well in targets:
        row, col = well_to_row_col(well)
        for field in range(1, sites + 1):
            written = list(channels)
            if incomplete_well is not None and well == incomplete_well and field == 1:
                written = written[:-1]
            for channel in written:
                path = image_dir / site_filename(row, col, field, channel)
                Image.fromarray(_pixels(row, col, field, channel, size)).save(path)

            if extra_files:
                for channel in (6, 7, 8):
                    path = image_dir / site_filename(row, col, field, channel)
                    Image.fromarray(_pixels(row, col, field, channel, size)).save(path)

    if extra_files:
        (image_dir / "Index.idx.xml").write_text("<Index />", encoding="utf-8")

    return image_dir
