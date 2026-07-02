"""Benchmark-aligned split strategies requiring external metadata.

These strategies depend on the CPJUMP1 reference metadata
(``data/reference/cpjump1/``) or the benchmark experiment metadata TSV.
Context loading lives in :mod:`benchmark.split_contexts`, the per-strategy
builders in :mod:`benchmark.split_strategies`; this module dispatches by strategy
name and builds the manifest / torch ``Subset`` objects.

Strategies:
    - ``cpjump1_official_representation``
    - ``cpjump1_official_gene_compound``
    - ``cellclip_cpjump_style``
"""

import logging

import pandas as pd
from torch.utils.data import Subset

from benchmark.split_contexts import (
    FALLBACK_METADATA_PATH,
    METADATA_PATH,
    OFFICIAL_SPLIT_METADATA_PATH,
    BenchmarkPlateContext,
    OfficialSplitContext,
    load_official_split_contexts,
    load_plate_contexts,
    resolve_metadata_path,
    resolve_official_split_metadata_path,
)
from benchmark.split_strategies import STRATEGY_BUILDERS
from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.perturbation import extract_plate_barcode

logger = logging.getLogger(__name__)

__all__ = [
    "METADATA_PATH",
    "FALLBACK_METADATA_PATH",
    "OFFICIAL_SPLIT_METADATA_PATH",
    "BenchmarkPlateContext",
    "OfficialSplitContext",
    "resolve_metadata_path",
    "resolve_official_split_metadata_path",
    "load_plate_contexts",
    "load_official_split_contexts",
    "build_split_manifest",
    "build_split_groups",
    "create_splits",
]


def _resolve_benchmark_split_indices(
    dataset: MorphoCLIPDataset,
    *,
    strategy: str,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve benchmark split indices by strategy name."""
    try:
        subsets_fn, _ = STRATEGY_BUILDERS[strategy]
    except KeyError:
        raise ValueError(f"Unknown benchmark split strategy: {strategy!r}") from None
    subsets = subsets_fn(dataset)
    return subsets["train"], subsets["validate"], subsets["test"]


def build_split_manifest(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a split manifest keyed by ``Metadata_Plate`` + ``Metadata_Well``."""
    del val_fraction, test_fraction, seed

    train_idx, val_idx, test_idx = _resolve_benchmark_split_indices(
        dataset,
        strategy=strategy,
    )
    subset_by_idx = {idx: "train" for idx in train_idx}
    subset_by_idx.update({idx: "validate" for idx in val_idx})
    subset_by_idx.update({idx: "test" for idx in test_idx})

    official_contexts: dict = {}
    try:
        official_contexts = load_official_split_contexts()
    except Exception:
        pass

    records: list[dict[str, str | int]] = []
    for idx, (plate, well, _) in enumerate(dataset.index_entries):
        subset = subset_by_idx.get(idx)
        if subset is None:
            continue

        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)
        context = official_contexts.get((barcode, well))
        records.append(
            {
                "split_strategy": strategy,
                "subset": subset,
                "Metadata_Plate": barcode,
                "Metadata_Well": well,
                "Metadata_broad_sample": info.broad_sample,
                "Metadata_cell_line": getattr(context, "cell_line", ""),
                "Metadata_experiment_type": getattr(context, "experiment_type", ""),
                "Metadata_timepoint": getattr(context, "timepoint", ""),
                "Metadata_timepoint_code": getattr(context, "timepoint_code", ""),
                "Metadata_target": getattr(context, "target", "") or info.gene or info.target_list,
            }
        )

    return pd.DataFrame.from_records(records)


def build_split_groups(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
) -> dict[str, list[int]]:
    """Build grouping keys for benchmark split strategies."""
    try:
        _, groups_fn = STRATEGY_BUILDERS[strategy]
    except KeyError:
        raise ValueError(f"Unknown benchmark split strategy: {strategy!r}") from None
    return groups_fn(dataset)


def create_splits(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test subsets using benchmark strategies."""
    train_idx, val_idx, test_idx = _resolve_benchmark_split_indices(
        dataset,
        strategy=strategy,
    )
    logger.info(
        "Benchmark split: train=%d, val=%d, test=%d (strategy=%s)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        strategy,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
