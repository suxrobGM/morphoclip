"""Build the split manifest CSV that the benchmark and CellCLIP trainer read back.

The required column set is ``{subset, Metadata_Plate, Metadata_Well}``. Manifests
already on disk behind the frozen CellCLIP results must keep loading, so those
three names are fixed.
"""

import logging

import pandas as pd

from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.perturbation import extract_plate_barcode
from morphoclip.splits.api import resolve_split_indices
from morphoclip.splits.contexts import load_official_split_contexts
from morphoclip.splits.strategies import SplitParams

logger = logging.getLogger(__name__)


def build_split_manifest(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
    params: SplitParams | None = None,
) -> pd.DataFrame:
    """Return one row per assigned well, keyed by plate and well.

    Args:
        dataset: Dataset whose wells to assign.
        strategy: Split strategy name.
        params: Passed through to the strategy; ignored by the benchmark ones.

    Returns:
        A DataFrame with the subset assignment plus the context columns the
        benchmark slices by.
    """
    train_idx, val_idx, test_idx = resolve_split_indices(dataset, strategy, params or SplitParams())
    subset_by_idx = dict.fromkeys(train_idx, "train")
    subset_by_idx.update(dict.fromkeys(val_idx, "validate"))
    subset_by_idx.update(dict.fromkeys(test_idx, "test"))

    # A missing or malformed file leaves the cell-line/timepoint columns empty,
    # which then silently drives the cellclip_cpjump_style split. Say so.
    official_contexts: dict = {}
    try:
        official_contexts = load_official_split_contexts()
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Official split metadata unavailable, context columns will be empty: %s", exc
        )

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
