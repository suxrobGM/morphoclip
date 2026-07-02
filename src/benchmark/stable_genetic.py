"""Genetic-modality evaluation steps for the stable CPJUMP1 benchmark.

Runs ORF/CRISPR replicability + matching and gene-compound (cross-modality)
matching, appending to a shared :class:`StableResults`. Consumes the
:class:`CompoundContext` produced by the compound step. Imports
copairs/scikit-learn transitively, so it is only imported when the benchmark runs.
"""

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
from benchmark.metrics import (
    evaluate_cross_modality_matching,
    evaluate_matching,
    evaluate_replicability,
)
from benchmark.profile_ops import concat_profiles
from benchmark.stable_compound import REPLICATE_FEATURE, CompoundContext
from benchmark.stable_config import BenchmarkParams
from benchmark.stable_helpers import (
    BatchCorrectionTransform,
    ProfileLoader,
    apply_batch_correction,
    load_profiles_for_plates,
    run_with_unpaired_guard,
)
from benchmark.stable_results import StableResults, _already_computed


def evaluate_genetic(
    results: StableResults,
    *,
    loader: ProfileLoader,
    params: BenchmarkParams,
    correction_transform: BatchCorrectionTransform | None,
    cell: str,
    cell_df: pd.DataFrame,
    compound: CompoundContext,
    count: int,
    batch_size: int,
    null_size: int,
) -> None:
    """Run genetic (ORF/CRISPR) replicability, matching, and gene-compound matching.

    Mutates *results* in place. Uses *compound* (the compound consensus context)
    for the cross-modality gene-compound matching step.
    """
    all_modality_2_experiments_df = cell_df[cell_df["Perturbation"] != compound.perturbation]
    for modality_2_perturbation in all_modality_2_experiments_df.Perturbation.unique():
        modality_2_experiments_df = all_modality_2_experiments_df.query(
            "Perturbation==@modality_2_perturbation"
        )
        for modality_2_timepoint in modality_2_experiments_df.Time.unique():
            modality_2_timepoint_df = modality_2_experiments_df.query(
                "Time==@modality_2_timepoint"
            )

            modality_2_df = load_profiles_for_plates(
                loader=loader,
                batch=params.batch,
                plates=list(modality_2_timepoint_df.Assay_Plate_Barcode.unique()),
                modality=modality_2_perturbation,
                attach_gene_target=True,
            )
            modality_2_df = filter_profiles_to_split_subset(
                modality_2_df,
                params.manifest,
                keep_negcon_on_selected_plates=True,
            )

            if modality_2_df.empty:
                print(
                    f"Skipping {modality_2_perturbation}_{cell}_"
                    f"{modality_2_timepoint}h - no data"
                )
                continue

            modality_2_df = apply_batch_correction(modality_2_df, correction_transform)
            modality_2_df = remove_empty_wells(modality_2_df)

            _time_2 = get_timepoint_label(modality_2_perturbation, modality_2_timepoint)
            description_2 = f"{modality_2_perturbation}_{cell}_{_time_2}"

            if not _already_computed(results.replicability_map, description_2):
                print(f"[{count}] Computing {description_2} replicability")

                result = run_with_unpaired_guard(
                    evaluate_replicability,
                    add_negcon_indicator(modality_2_df),
                    null_size=null_size,
                    batch_size=batch_size,
                    copairs_mode="stable",
                )
                if result.empty:
                    print(f"Skipping {description_2} replicability - no valid pairs")
                    continue
                results.append(
                    "replicability_map",
                    "replicability_fr",
                    result=result,
                    group_cols=[REPLICATE_FEATURE],
                    null_size=null_size,
                    metadata={
                        "Description": description_2,
                        "Modality": modality_2_perturbation,
                        "Cell": cell,
                        "time": _time_2,
                        "timepoint": modality_2_timepoint,
                    },
                )

            modality_2_df = remove_negcon_wells(modality_2_df)

            modality_2_consensus_df = compute_consensus(modality_2_df, "Metadata_broad_sample")

            replicable_genes = list(
                results.replicability_map[
                    (results.replicability_map.Description == description_2)
                    & results.replicability_map.above_q_threshold
                ][REPLICATE_FEATURE]
            )
            modality_2_consensus_df = filter_replicable(
                modality_2_consensus_df,
                replicable_genes,
                id_col=REPLICATE_FEATURE,
            )

            # Filter out reagents without a sister guide.
            gene_counts = modality_2_consensus_df["Metadata_gene"].value_counts()
            genes_without_sister = gene_counts[gene_counts == 1].index.to_list()

            modality_2_consensus_for_matching_df = modality_2_consensus_df.loc[
                ~modality_2_consensus_df["Metadata_gene"].isin(genes_without_sister)
            ].reset_index(drop=True)

            if modality_2_perturbation == "crispr":
                if not _already_computed(results.matching_map, description_2):
                    print(f"[{count}] Computing {description_2} matching")

                    result = run_with_unpaired_guard(
                        evaluate_matching,
                        modality_2_consensus_for_matching_df,
                        target_col="Metadata_matching_target",
                        use_abs=False,
                        multilabel=False,
                        null_size=null_size,
                        batch_size=batch_size,
                        copairs_mode="stable",
                    )
                    if result.empty:
                        print(f"Skipping {description_2} matching - no valid pairs")
                        continue

                    results.append(
                        "matching_map",
                        "matching_fr",
                        result=result,
                        group_cols=["Metadata_matching_target"],
                        null_size=null_size,
                        metadata={
                            "Description": description_2,
                            "Modality": modality_2_perturbation,
                            "Cell": cell,
                            "time": _time_2,
                            "timepoint": modality_2_timepoint,
                        },
                    )

            _evaluate_cross_modality(
                results,
                compound=compound,
                modality_2_consensus_df=modality_2_consensus_df,
                modality_2_perturbation=modality_2_perturbation,
                cell=cell,
                time_2=_time_2,
                modality_2_timepoint=modality_2_timepoint,
                count=count,
                batch_size=batch_size,
                null_size=null_size,
            )


def _evaluate_cross_modality(
    results: StableResults,
    *,
    compound: CompoundContext,
    modality_2_consensus_df: pd.DataFrame,
    modality_2_perturbation: str,
    cell: str,
    time_2: str,
    modality_2_timepoint: int,
    count: int,
    batch_size: int,
    null_size: int,
) -> None:
    """Gene-compound (cross-modality) matching for one genetic timepoint.

    Restricts the compound consensus to targets perturbed in this genetic
    modality, then runs cross-modality matching and appends the result. Returns
    early (no-op) when there are no overlapping targets or no valid pairs.
    """
    modality_1_consensus_df = compound.modality_1_consensus_df
    modality_1_perturbation = compound.perturbation
    _time = compound.time_label
    modality_1_timepoint = compound.timepoint

    perturbed_genes = list(set(modality_2_consensus_df.Metadata_matching_target))
    modality_1_targets_df = (
        modality_1_consensus_df[["Metadata_broad_sample", "Metadata_matching_target"]]
        .copy()
        .explode("Metadata_matching_target")
    )

    modality_1_filtered_genes_df = (
        modality_1_targets_df[
            modality_1_targets_df["Metadata_matching_target"].isin(perturbed_genes)
        ]
        .reset_index(drop=True)
        .groupby(["Metadata_broad_sample"])
        .Metadata_matching_target.apply(list)
        .reset_index()
    )

    modality_1_consensus_filtered_df = modality_1_consensus_df.drop(
        columns=["Metadata_matching_target"]
    ).merge(
        modality_1_filtered_genes_df,
        on="Metadata_broad_sample",
        how="inner",
    )

    if modality_1_consensus_filtered_df.empty:
        print("Skipping gene-compound matching - no overlapping targets")
        return

    description_cross = (
        f"{modality_1_perturbation}_{cell}_{_time}"
        f"-{modality_2_perturbation}_{cell}_{time_2}"
    )
    print(f"[{count}] Computing {description_cross} matching")

    modality_1_modality_2_df = concat_profiles(
        modality_1_consensus_filtered_df, modality_2_consensus_df
    )

    result = run_with_unpaired_guard(
        evaluate_cross_modality_matching,
        modality_1_modality_2_df,
        target_col="Metadata_matching_target",
        null_size=null_size,
        batch_size=batch_size,
        copairs_mode="stable",
    )
    if result.empty:
        print(f"Skipping {description_cross} matching - no valid pairs")
        return

    results.append(
        "gene_compound_matching_map",
        "gene_compound_matching_fr",
        result=result,
        group_cols=["Metadata_matching_target"],
        null_size=null_size,
        metadata={
            "Description": description_cross,
            "Modality1": modality_1_perturbation,
            "Modality2": modality_2_perturbation,
            "Cell": cell,
            "time1": _time,
            "time2": time_2,
            "timepoint1": modality_1_timepoint,
            "timepoint2": modality_2_timepoint,
        },
    )
