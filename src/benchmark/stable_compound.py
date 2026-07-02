"""Compound-modality evaluation step for the stable CPJUMP1 benchmark.

Loads and corrects compound profiles, runs replicability + target matching, and
appends to a shared :class:`StableResults`. Produces the :class:`CompoundContext`
the genetic step consumes. Imports copairs/scikit-learn transitively, so it is
only imported when the benchmark runs.
"""

from dataclasses import dataclass

import pandas as pd

from benchmark.data import (
    add_negcon_indicator,
    compute_consensus,
    filter_profiles_to_split_subset,
    filter_replicable,
    get_timepoint_label,
    remove_empty_wells,
    remove_negcon_wells,
)
from benchmark.metrics import evaluate_matching, evaluate_replicability
from benchmark.stable_config import BenchmarkParams
from benchmark.stable_helpers import (
    BatchCorrectionTransform,
    ProfileLoader,
    apply_batch_correction,
    load_profiles_for_plates,
    run_with_unpaired_guard,
)
from benchmark.stable_results import StableResults

REPLICATE_FEATURE = "Metadata_broad_sample"
MODALITY_1_PERTURBATION = "compound"


@dataclass
class CompoundContext:
    """Compound-modality outputs the genetic step depends on."""

    modality_1_consensus_df: pd.DataFrame
    time_label: str
    timepoint: int
    perturbation: str


def evaluate_compound(
    results: StableResults,
    *,
    loader: ProfileLoader,
    params: BenchmarkParams,
    correction_transform: BatchCorrectionTransform | None,
    target1_metadata: pd.DataFrame,
    cell: str,
    modality_1_experiments_df: pd.DataFrame,
    modality_1_timepoint: int,
    count: int,
    batch_size: int,
    null_size: int,
) -> CompoundContext | None:
    """Run compound replicability + target matching for one cell/timepoint.

    Returns the compound consensus context for the genetic step, or ``None`` when
    the modality has no data or no replicable pairs (caller should skip).
    """
    modality_1_perturbation = MODALITY_1_PERTURBATION
    modality_1_timepoint_df = modality_1_experiments_df.query("Time==@modality_1_timepoint")
    modality_1_df = load_profiles_for_plates(
        loader=loader,
        batch=params.batch,
        plates=list(modality_1_timepoint_df.Assay_Plate_Barcode.unique()),
        modality=modality_1_perturbation,
    )
    modality_1_df = filter_profiles_to_split_subset(
        modality_1_df,
        params.manifest,
        keep_negcon_on_selected_plates=True,
    )

    if modality_1_df.empty:
        print(f"Skipping {modality_1_perturbation}_{cell}_{modality_1_timepoint}h - no data")
        return None

    modality_1_df = apply_batch_correction(modality_1_df, correction_transform)
    modality_1_df[REPLICATE_FEATURE] = modality_1_df[REPLICATE_FEATURE].fillna("DMSO")
    modality_1_df = remove_empty_wells(modality_1_df)

    _time = get_timepoint_label(modality_1_perturbation, modality_1_timepoint)
    description = f"{modality_1_perturbation}_{cell}_{_time}"
    print(f"[{count}] Computing {description} replicability")

    result = run_with_unpaired_guard(
        evaluate_replicability,
        add_negcon_indicator(modality_1_df),
        null_size=null_size,
        batch_size=batch_size,
        copairs_mode="stable",
    )
    if result.empty:
        print(f"Skipping {description} replicability - no valid pairs")
        return None
    results.append(
        "replicability_map",
        "replicability_fr",
        result=result,
        group_cols=[REPLICATE_FEATURE],
        null_size=null_size,
        metadata={
            "Description": description,
            "Modality": modality_1_perturbation,
            "Cell": cell,
            "time": _time,
            "timepoint": modality_1_timepoint,
        },
    )

    # --- Compound matching ---
    modality_1_df = remove_negcon_wells(modality_1_df)
    modality_1_consensus_df = compute_consensus(modality_1_df, REPLICATE_FEATURE)

    replicable_compounds = list(
        results.replicability_map[
            (results.replicability_map.Description == description)
            & results.replicability_map.above_q_threshold
        ][REPLICATE_FEATURE]
    )
    modality_1_consensus_df = filter_replicable(
        modality_1_consensus_df,
        replicable_compounds,
        id_col=REPLICATE_FEATURE,
    )

    modality_1_consensus_df = (
        modality_1_consensus_df.merge(target1_metadata, on="Metadata_broad_sample", how="left")
        .assign(Metadata_matching_target=lambda x: x.Metadata_target_list.str.split("|"))
        .drop(["Metadata_target_list"], axis=1)
    )

    print(f"[{count}] Computing {description} matching")

    result = run_with_unpaired_guard(
        evaluate_matching,
        modality_1_consensus_df,
        target_col="Metadata_matching_target",
        use_abs=True,
        multilabel=True,
        null_size=null_size,
        batch_size=batch_size,
        copairs_mode="stable",
    )
    if result.empty:
        print(f"Skipping {description} matching - no valid pairs")
    else:
        results.append(
            "matching_map",
            "matching_fr",
            result=result,
            group_cols=["Metadata_matching_target"],
            null_size=null_size,
            metadata={
                "Description": description,
                "Modality": modality_1_perturbation,
                "Cell": cell,
                "time": _time,
                "timepoint": modality_1_timepoint,
            },
        )

    return CompoundContext(
        modality_1_consensus_df=modality_1_consensus_df,
        time_label=_time,
        timepoint=modality_1_timepoint,
        perturbation=modality_1_perturbation,
    )
