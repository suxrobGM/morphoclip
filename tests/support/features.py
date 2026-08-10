"""Builders for a fake pre-extracted feature cache and its text sidecar."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import yaml

from morphoclip.data.metadata import MetadataIndex
from morphoclip.training.config import EMBED_DIM, INPUT_CHANNELS, TEXT_INPUT_DIM

BATCH = "2020_11_04_CPJUMP1"


def well_to_row_col(well: str) -> tuple[int, int]:
    """Turn ``"A01"`` into ``(1, 1)``."""
    return ord(well[0].upper()) - 64, int(well[1:])


def _tensor(plate: str, well: str, site: int, channels: int, dim: int) -> torch.Tensor:
    """Deterministic per-site features, so a failure reproduces."""
    generator = torch.Generator().manual_seed(hash((plate, well, site)) % (2**31))
    return torch.randn(channels, dim, generator=generator)


def make_feature_root(
    root: Path,
    plates: Mapping[str, Sequence[str]],
    *,
    sites: int | Mapping[str, int] = 1,
    channels: int = INPUT_CHANNELS,
    dim: int = EMBED_DIM,
) -> Path:
    """Write a ``<root>/<plate>/r{rr}c{cc}f{ff}.pt`` tree of fake CLS tokens.

    Args:
        root: Feature root to create.
        plates: ``{plate_barcode: [well, ...]}``.
        sites: Fields per well, or ``{well: n_fields}`` for ragged wells.
        channels: Fluorescence channels per site.
        dim: Feature width. Must stay ``EMBED_DIM`` for anything that builds a
            real image encoder, since that width is a module constant.

    Returns:
        ``root``.
    """
    for plate, wells in plates.items():
        plate_dir = root / plate
        plate_dir.mkdir(parents=True, exist_ok=True)
        for well in wells:
            row, col = well_to_row_col(well)
            n_sites = sites if isinstance(sites, int) else sites[well]
            for site in range(1, n_sites + 1):
                path = plate_dir / f"r{row:02d}c{col:02d}f{site:02d}.pt"
                torch.save(_tensor(plate, well, site, channels, dim), path)
    return root


def write_well_features(
    root: Path, plate: str, well: str, *, sites: int = 1, dim: int = EMBED_DIM
) -> Path:
    """One well of :func:`make_feature_root`, for tests that add wells one at a time."""
    return make_feature_root(root, {plate: [well]}, sites=sites, dim=dim)


def write_dataset_yml(path: Path, *, metadata_dir: Path, batch: str = BATCH) -> Path:
    """Write a ``configs/dataset.yml``-shaped file for ``MetadataIndex.from_config``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"cpjump": {"batch": batch, "local": {"metadata": str(metadata_dir)}}}),
        encoding="utf-8",
    )
    return path


def write_text_cache(
    path: Path,
    metadata: MetadataIndex,
    plates: Mapping[str, Sequence[str]],
    *,
    dim: int = TEXT_INPUT_DIM,
) -> Path:
    """Write a text cache covering every well in *plates*.

    Mirrors ``utils.caching`` on disk: raw BERT-width embeddings keyed by
    ``broad_sample``, which is what ``lookup_text_embeddings`` indexes with.
    """
    ids = sorted(
        {
            metadata.lookup(plate, well).broad_sample
            for plate, wells in plates.items()
            for well in wells
        }
    )
    generator = torch.Generator().manual_seed(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": torch.randn(len(ids), dim, generator=generator),
            "perturbation_ids": ids,
            "id_to_idx": {pid: i for i, pid in enumerate(ids)},
            "raw": True,
        },
        path,
    )
    return path
