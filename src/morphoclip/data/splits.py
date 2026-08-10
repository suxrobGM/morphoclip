"""Train/val/test splitting for MorphoCLIP datasets.

The ``pert_type`` strategy uses only local metadata (``data/metadata/``).
Benchmark-aligned strategies (``cpjump1_official_*``, ``cellclip_cpjump_style``)
live in ``benchmark.splits`` and are not handled here.
"""

import hashlib
import logging
from collections import defaultdict

from torch.utils.data import Subset

from morphoclip.data.dataset import MorphoCLIPDataset, MorphoCLIPSample
from morphoclip.data.perturbation import (
    extract_plate_barcode,
    is_control_or_empty,
)

logger = logging.getLogger(__name__)


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
    seed: int = 42,
) -> tuple[Subset[MorphoCLIPSample], Subset[MorphoCLIPSample], Subset[MorphoCLIPSample]]:
    """Split dataset into train/val/test subsets.

    Args:
        dataset: The full dataset.
        strategy: ``"pert_type"`` — stratified split across all perturbation
            types (compounds, CRISPR, ORF).  Uses only local metadata.
            For benchmark strategies, use ``benchmark.splits`` directly.
        val_fraction: Fraction of broad_samples for validation. The same
            fraction is used for test.
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
