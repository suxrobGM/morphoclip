"""PyTorch Dataset classes for MorphoCLIP training.

Provides Dataset implementations for pre-extracted DINOv3 features
(default training path) and resized image tensors (LoRA experiments).
Handles well-level aggregation and text description pairing.

Each training sample is a *well*: all imaging sites within the well are
stacked into a single tensor and paired with the well's text description.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset

from morphoclip.data.image_loader import FEATURE_PATTERN
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import (
    PerturbationInfo,
    PerturbationType,
    extract_plate_barcode,
    generate_text,
    well_from_row_col,
)

logger = logging.getLogger(__name__)


class MorphoCLIPSample:
    """A single training sample (one well).

    Attributes:
        features: Stacked site features or tensors.
            Feature mode: ``(num_sites, num_channels, hidden_dim)``
            e.g. ``(9, 5, 1024)``.
            Tensor mode: ``(num_sites, num_channels, H, W)``
            e.g. ``(9, 5, 384, 384)``.
        text: Text description of the perturbation.
        plate: Plate barcode.
        well: Well position string (e.g. ``"A01"``).
        pert_info: Full perturbation metadata.
    """

    __slots__ = ("features", "text", "plate", "well", "pert_info")

    def __init__(
        self,
        features: torch.Tensor,
        text: str,
        plate: str,
        well: str,
        pert_info: PerturbationInfo,
    ) -> None:
        self.features = features
        self.text = text
        self.plate = plate
        self.well = well
        self.pert_info = pert_info


class MorphoCLIPDataset(Dataset[MorphoCLIPSample]):
    """Dataset for MorphoCLIP contrastive training.

    Loads pre-extracted DINOv3 features or resized image tensors,
    paired with text descriptions from the metadata index.
    Each sample is a well: all sites within the well are aggregated.

    Usage::

        metadata = MetadataIndex.from_config(Path("configs/dataset.yml"))
        dataset = MorphoCLIPDataset(
            feature_dir=Path("data/features"),
            metadata=metadata,
            plates=["BR00116991"],
        )
        sample = dataset[0]
    """

    def __init__(
        self,
        feature_dir: Path,
        metadata: MetadataIndex,
        plates: list[str],
        mode: Literal["features", "tensors"] = "features",
        text_level: str = "full",
        exclude_controls: bool = False,
        pert_types: set[PerturbationType] | None = None,
        max_sites_per_well: int | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            feature_dir: Root directory containing per-plate subdirectories
                with ``.pt`` files (``data/features/`` or ``data/tensors/``).
            metadata: MetadataIndex for text description lookup.
            plates: List of plate barcodes to include.
            mode: ``"features"`` for pre-extracted CLS tokens,
                ``"tensors"`` for resized image tensors.
            text_level: Text granularity
                (``"name_only"``, ``"name_target"``, ``"full"``).
            exclude_controls: If ``True``, exclude negcon/poscon wells.
            pert_types: If provided, include only these perturbation types.
            max_sites_per_well: Cap sites per well (random subset).
        """
        self._feature_dir = feature_dir
        self._metadata = metadata
        self._plates = plates
        self._mode = mode
        self._text_level = text_level
        self._exclude_controls = exclude_controls
        self._pert_types = pert_types
        self._max_sites = max_sites_per_well

        # Internal index: list of (plate_barcode, well, [site_paths])
        self._index: list[tuple[str, str, list[Path]]] = []
        self._build_index()

        # In-memory cache: path -> tensor (populated by preload())
        self._cache: dict[Path, torch.Tensor] = {}

    def _build_index(self) -> None:
        """Scan feature directories and group .pt files by well."""
        for plate in self._plates:
            plate_dir = self._feature_dir / plate
            if not plate_dir.exists():
                logger.warning("Plate directory not found: %s", plate_dir)
                continue

            well_files: dict[str, list[Path]] = defaultdict(list)
            for pt_file in sorted(plate_dir.glob("*.pt")):
                match = FEATURE_PATTERN.match(pt_file.name)
                if match is None:
                    continue
                row, col = int(match["row"]), int(match["col"])
                well = well_from_row_col(row, col)
                well_files[well].append(pt_file)

            barcode = extract_plate_barcode(plate)
            for well in sorted(well_files.keys()):
                info = self._metadata.lookup(barcode, well)

                if self._exclude_controls and info.pert_type in (
                    PerturbationType.NEGCON,
                    PerturbationType.POSCON,
                ):
                    continue

                if self._pert_types is not None and info.pert_type not in self._pert_types:
                    continue

                self._index.append((plate, well, well_files[well]))

        logger.info(
            "Built dataset index: %d wells across %d plates",
            len(self._index),
            len(self._plates),
        )

    def preload(self, *, indices: set[int] | None = None) -> None:
        """Load feature tensors into memory for faster training.

        Args:
            indices: If provided, only preload wells at these dataset
                indices (e.g. train+val).  Otherwise preloads everything.
        """
        if self._cache:
            return
        entries = self._index
        if indices is not None:
            entries = [self._index[i] for i in indices if i < len(self._index)]
        paths = sorted({p for _, _, site_paths in entries for p in site_paths})

        logger.info("Preloading %d feature files into memory...", len(paths))
        for i, p in enumerate(paths):
            # .clone() copies the 20 KB tensor and lets GC free the
            # ~960 KB deserialized buffer from torch.load
            self._cache[p] = torch.load(p, weights_only=True).clone()
            if (i + 1) % 20_000 == 0:
                logger.info("  %d / %d loaded", i + 1, len(paths))

        size_gb = sum(t.nbytes for t in self._cache.values()) / 1024**3
        logger.info("Preload complete: %d files, %.1f GB", len(paths), size_gb)

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Load a tensor from cache or disk."""
        if self._cache:
            return self._cache[path]
        return torch.load(path, weights_only=True)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> MorphoCLIPSample:
        """Load a well's features and pair with text description."""
        plate, well, site_paths = self._index[idx]
        barcode = extract_plate_barcode(plate)

        paths = site_paths
        if self._max_sites is not None and len(paths) > self._max_sites:
            indices = torch.randperm(len(paths))[: self._max_sites].sort().values
            paths = [paths[i] for i in indices]

        site_tensors = [self._load_tensor(p) for p in paths]
        features = torch.stack(site_tensors, dim=0)

        info = self._metadata.lookup(barcode, well)
        text = generate_text(info, level=self._text_level)

        return MorphoCLIPSample(
            features=features,
            text=text,
            plate=plate,
            well=well,
            pert_info=info,
        )

    @property
    def metadata(self) -> MetadataIndex:
        """The metadata index used by this dataset."""
        return self._metadata

    @property
    def index_entries(self) -> list[tuple[str, str, list[Path]]]:
        """Access to the internal index for splitting."""
        return self._index


def collate_fn(
    batch: list[MorphoCLIPSample],
) -> dict[str, torch.Tensor | list[str] | list[PerturbationInfo]]:
    """Custom collate function for variable site counts per well.

    Pads to the maximum number of sites in the batch and provides
    a boolean mask.

    Returns:
        Dict with keys:
            - ``features``: ``(B, max_sites, C, D)`` or ``(B, max_sites, C, H, W)``
            - ``site_mask``: ``(B, max_sites)`` boolean, ``True`` for real sites
            - ``text``: list of strings (length B)
            - ``plates``: list of plate strings (length B)
            - ``wells``: list of well strings (length B)
            - ``pert_info``: list of PerturbationInfo (length B)
    """
    max_sites = max(s.features.shape[0] for s in batch)
    feature_shape = batch[0].features.shape[1:]  # (C, D) or (C, H, W)

    padded_features = torch.zeros(len(batch), max_sites, *feature_shape)
    site_mask = torch.zeros(len(batch), max_sites, dtype=torch.bool)

    texts: list[str] = []
    plates: list[str] = []
    wells: list[str] = []
    pert_infos: list[PerturbationInfo] = []

    for i, sample in enumerate(batch):
        n_sites = sample.features.shape[0]
        padded_features[i, :n_sites] = sample.features
        site_mask[i, :n_sites] = True
        texts.append(sample.text)
        plates.append(sample.plate)
        wells.append(sample.well)
        pert_infos.append(sample.pert_info)

    return {
        "features": padded_features,
        "site_mask": site_mask,
        "text": texts,
        "plates": plates,
        "wells": wells,
        "pert_info": pert_infos,
    }
