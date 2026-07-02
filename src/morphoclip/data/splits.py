"""Train/val/test splitting for MorphoCLIP datasets.

The ``pert_type`` strategy uses only local metadata (``data/metadata/``).
Benchmark-aligned strategies (``cpjump1_official_*``, ``cellclip_cpjump_style``)
live in ``benchmark.splits`` and are not handled here.
"""

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd
from torch.utils.data import Subset

from morphoclip.data.dataset import MorphoCLIPDataset, MorphoCLIPSample
from morphoclip.data.perturbation import (
    extract_plate_barcode,
    is_control_or_empty,
)

logger = logging.getLogger(__name__)


# --- pert_type strategy (local metadata only) ---


def _build_pert_type_subsets(
    dataset: MorphoCLIPDataset,
    *,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split wells stratified by perturbation type using local metadata.

    All perturbation types (compound, CRISPR, ORF) are distributed across
    train/val/test so that each split sees every modality.  Wells sharing
    the same ``broad_sample`` always land in the same split.

    Args:
        dataset: MorphoCLIP dataset with populated metadata.
        val_fraction: Fraction of ``broad_sample`` groups assigned to
            validate.  An equal fraction goes to test; the rest to train.
        seed: Seed for deterministic splitting.

    Returns:
        Dict with ``"train"``, ``"validate"``, ``"test"`` index lists.
    """
    # Group dataset indices by broad_sample
    sample_to_indices: dict[str, list[int]] = defaultdict(list)

    for i, (plate, well, _) in enumerate(dataset.index_entries):
        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)

        if is_control_or_empty(info):
            continue

        sample_to_indices[info.broad_sample].append(i)

    # Deterministically assign each broad_sample to a split
    subsets: dict[str, list[int]] = {"train": [], "validate": [], "test": []}
    test_fraction = val_fraction  # equal val and test fractions

    for sample in sorted(sample_to_indices.keys()):
        h = hashlib.md5(f"{seed}:{sample}".encode()).hexdigest()
        fraction = int(h[:8], 16) / 0xFFFFFFFF
        if fraction < val_fraction:
            subset = "validate"
        elif fraction < val_fraction + test_fraction:
            subset = "test"
        else:
            subset = "train"
        subsets[subset].extend(sample_to_indices[sample])

    logger.info(
        "pert_type split (mixed): train=%d, validate=%d, test=%d",
        len(subsets["train"]),
        len(subsets["validate"]),
        len(subsets["test"]),
    )

    return subsets


def _build_pert_type_groups(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """Build grouping keys for the pert_type strategy.

    Returns:
        Mapping ``"pert_type::broad_sample" -> dataset indices``.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, (plate, well, _) in enumerate(dataset.index_entries):
        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)
        if is_control_or_empty(info):
            continue
        group_key = f"{info.pert_type.value}::{info.broad_sample}"
        grouped[group_key].append(i)
    return dict(grouped)


# --- Public API ---


def build_split_groups(
    dataset: MorphoCLIPDataset,
    strategy: str = "pert_type",
) -> dict[str, list[int]]:
    """Build grouping keys used by split strategies.

    Args:
        dataset: Dataset whose ``index_entries`` define the split units.
        strategy: ``"pert_type"`` (other strategies in ``benchmark.splits``).

    Returns:
        Mapping ``group_key -> dataset indices``.
    """
    if strategy == "pert_type":
        return _build_pert_type_groups(dataset)
    raise ValueError(
        f"Unknown split strategy {strategy!r}. Benchmark strategies are in benchmark.splits."
    )


def build_split_manifest(
    dataset: MorphoCLIPDataset,
    strategy: str = "pert_type",
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a split manifest keyed by ``Metadata_Plate`` + ``Metadata_Well``."""
    train_idx, val_idx, test_idx = _resolve_split_indices(
        dataset,
        strategy=strategy,
        val_fraction=val_fraction,
        seed=seed,
    )
    subset_by_idx = {idx: "train" for idx in train_idx}
    subset_by_idx.update({idx: "validate" for idx in val_idx})
    subset_by_idx.update({idx: "test" for idx in test_idx})

    records: list[dict[str, str | int]] = []
    for idx, (plate, well, _) in enumerate(dataset.index_entries):
        subset = subset_by_idx.get(idx)
        if subset is None:
            continue

        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)
        records.append(
            {
                "split_strategy": strategy,
                "subset": subset,
                "Metadata_Plate": barcode,
                "Metadata_Well": well,
                "Metadata_broad_sample": info.broad_sample,
                "Metadata_pert_type": info.pert_type.value,
                "Metadata_target": info.gene or info.target_list,
            }
        )

    return pd.DataFrame.from_records(records)


def save_split_manifest(
    dataset: MorphoCLIPDataset,
    output_path: str | Path,
    strategy: str = "pert_type",
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Build and save a split manifest CSV."""
    manifest = build_split_manifest(
        dataset,
        strategy=strategy,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def _resolve_split_indices(
    dataset: MorphoCLIPDataset,
    *,
    strategy: str,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if strategy == "pert_type":
        subsets = _build_pert_type_subsets(
            dataset,
            val_fraction=val_fraction,
            seed=seed,
        )
        return subsets["train"], subsets["validate"], subsets["test"]
    raise ValueError(
        f"Unknown split strategy {strategy!r}. Benchmark strategies are in benchmark.splits."
    )


def create_splits(
    dataset: MorphoCLIPDataset,
    strategy: str = "pert_type",
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Subset[MorphoCLIPSample], Subset[MorphoCLIPSample], Subset[MorphoCLIPSample]]:
    """Split dataset into train/val/test subsets.

    Args:
        dataset: The full dataset.
        strategy: ``"pert_type"`` — stratified split across all perturbation
            types (compounds, CRISPR, ORF).  Uses only local metadata.
            For benchmark strategies, use ``benchmark.splits`` directly.
        val_fraction: Fraction of broad_samples for validation (equal
            fraction used for test).
        test_fraction: Unused (val_fraction controls both val and test).
        seed: Random seed for deterministic compound splitting.

    Returns:
        ``(train, val, test)`` tuple of ``Subset`` objects.
    """
    train_idx, val_idx, test_idx = _resolve_split_indices(
        dataset,
        strategy=strategy,
        val_fraction=val_fraction,
        seed=seed,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d (strategy=%s)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        strategy,
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )
