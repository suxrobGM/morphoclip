"""Stable benchmark evaluation aligned to the 2024 Chandrasekaran reference pipeline.

This module holds the full orchestration for the CPJUMP1 "stable copairs" benchmark.
The CLI wrapper (`morphoclip benchmark`) is a thin adapter over
:func:`run_stable_benchmark`.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

from benchmark.data import (
    ProfileLoader,
    add_negcon_indicator,
    compute_consensus,
    filter_experiment_metadata_to_split_subset,
    filter_profiles_to_split_subset,
    filter_replicable,
    get_timepoint_label,
    load_split_manifest,
    normalize_subset_label,
    remove_empty_wells,
    remove_negcon_wells,
)
from benchmark.metrics import (
    evaluate_cross_modality_matching,
    evaluate_matching,
    evaluate_replicability,
)
from benchmark.stable_helpers import (
    apply_batch_correction,
    compute_map_and_fr,
    concat_profiles,
    fit_batch_correction,
    load_profiles_for_plates,
    plot_cross_modality_barplot,
    plot_matching_barplot,
    plot_matching_fr_faceted,
    plot_matching_map_boxplot,
    plot_replicability_barplot,
    plot_replicability_fr_faceted,
    plot_replicability_map_boxplot,
    run_with_unpaired_guard,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG_PATH = Path("configs/benchmark.yml")
TIMELINE_CHOICES = ("short", "long")


def load_benchmark_config(path: Path) -> dict:
    """Load benchmark defaults from YAML if present."""
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark config must be a mapping: {path}")

    config = raw.get("benchmark", raw)
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config section must be a mapping: {path}")
    return config


def normalize_timelines(value) -> list[str]:
    """Normalize timeline selection to a validated list of labels."""
    if value is None:
        return list(TIMELINE_CHOICES)

    values = [value] if isinstance(value, str) else list(value)

    normalized: list[str] = []
    for item in values:
        label = str(item).strip().lower()
        if label not in TIMELINE_CHOICES:
            choices = ", ".join(TIMELINE_CHOICES)
            raise ValueError(f"Invalid timeline {item!r}; expected one of: {choices}")
        if label not in normalized:
            normalized.append(label)

    return normalized or list(TIMELINE_CHOICES)


def resolve_path(project_root: Path, value: str | None) -> Path | None:
    """Resolve an optional CLI/config path relative to the project root."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def run_stable_benchmark(
    *,
    config: Path = CONFIG_PATH,
    profiles_dir: str | None = None,
    output_dir: str | None = None,
    batch: str | None = None,
    test_mode: bool | None = None,
    cell_filter: str | None = None,
    batch_correction: bool | None = None,
    pca_kernel: str | None = None,
    pca_n_components: int | None = None,
    split_manifest: str | None = None,
    split_subset: str | None = None,
    timelines: list[str] | None = None,
) -> None:
    """Run the CPJUMP1 benchmark evaluation (stable copairs mode).

    All arguments are optional CLI overrides; unset values fall back to the
    ``benchmark`` section of the YAML config at *config*.
    """
    cfg = load_benchmark_config(config)

    profiles_dir_arg = profiles_dir or cfg.get("profiles_dir", "data/profiles")
    output_dir_arg = output_dir or cfg.get("output_dir", "output/benchmark")
    resolved_batch = batch or cfg.get("batch", "2020_11_04_CPJUMP1")
    resolved_test_mode = test_mode if test_mode is not None else cfg.get("test_mode", False)
    resolved_cell_filter = cell_filter or cfg.get("cell_filter")
    resolved_batch_correction = (
        batch_correction if batch_correction is not None else cfg.get("batch_correction", False)
    )
    resolved_pca_kernel = pca_kernel or cfg.get("pca_kernel", "linear")
    resolved_pca_n_components = int(pca_n_components or cfg.get("pca_n_components", 500))
    split_manifest_arg = split_manifest or cfg.get("split_manifest")
    split_subset_arg = split_subset or cfg.get("split_subset")
    resolved_timelines = normalize_timelines(
        timelines if timelines is not None else cfg.get("timelines")
    )

    project_root = Path(__file__).resolve().parents[2]
    profiles_path = project_root / profiles_dir_arg
    output_path = project_root / output_dir_arg
    figures_dir = output_path / "figures"
    tables_dir = output_path / "tables"
    split_manifest_path = resolve_path(project_root, split_manifest_arg)
    subset = None
    manifest = None

    output_path.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    if split_manifest_path is not None:
        if split_subset_arg is None:
            raise ValueError("--split-subset is required when --split-manifest is provided")
        subset = normalize_subset_label(split_subset_arg)
        manifest = load_split_manifest(split_manifest_path, subset)
    elif split_subset_arg is not None:
        raise ValueError("--split-subset requires --split-manifest")

    exp_meta_path = project_root / "output/benchmark/input/experiment-metadata.tsv"
    target_ann_path = (
        project_root
        / "output/benchmark/input/JUMP-Target-1_compound_metadata_additional_annotations.tsv"
    )

    if not exp_meta_path.exists():
        print(f"ERROR: Missing experiment metadata: {exp_meta_path}")
        sys.exit(1)
    if not target_ann_path.exists():
        print(f"ERROR: Missing target annotations: {target_ann_path}")
        sys.exit(1)

    experiment_df = (
        pd.read_csv(exp_meta_path, sep="\t")
        .query(f"Batch=='{resolved_batch}'")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )
    experiment_df = experiment_df.drop(
        experiment_df[
            (experiment_df.Perturbation == "compound") & (experiment_df.Cell_line == "Cas9")
        ].index
    )
    experiment_df = filter_experiment_metadata_to_split_subset(experiment_df, manifest)
    experiment_df["timeline"] = experiment_df.apply(
        lambda row: get_timepoint_label(row["Perturbation"], row["Time"]),
        axis=1,
    )
    experiment_df = experiment_df[experiment_df["timeline"].isin(resolved_timelines)].reset_index(
        drop=True
    )

    if experiment_df.empty:
        print(
            "ERROR: No experiments matched the selected timeline(s): "
            f"{', '.join(resolved_timelines)} for batch {resolved_batch}"
        )
        sys.exit(1)

    target1_metadata = pd.read_csv(
        target_ann_path, sep="\t", usecols=["broad_sample", "target_list"]
    ).rename(
        columns={"broad_sample": "Metadata_broad_sample", "target_list": "Metadata_target_list"}
    )

    if resolved_test_mode:
        print("TEST MODE: Using limited data")
        experiment_df = experiment_df.head(8)
        print(experiment_df)

    if resolved_cell_filter:
        experiment_df = experiment_df.query(f"Cell_type=='{resolved_cell_filter}'")

    if experiment_df.empty:
        print("ERROR: No experiments remained after applying filters.")
        sys.exit(1)

    replicate_feature = "Metadata_broad_sample"
    batch_size = 100000 if not resolved_test_mode else 20000
    null_size = 100000 if not resolved_test_mode else 10000
    loader = ProfileLoader(profiles_path)

    replicability_map_df = pd.DataFrame()
    replicability_fr_df = pd.DataFrame()
    matching_map_df = pd.DataFrame()
    matching_fr_df = pd.DataFrame()
    gene_compound_matching_map_df = pd.DataFrame()
    gene_compound_matching_fr_df = pd.DataFrame()

    print("=" * 60)
    print("CPJUMP1 Benchmark Evaluation (Stable Mode)")
    print("=" * 60)
    print(f"Profiles: {profiles_path}")
    print(f"Batch: {resolved_batch}")
    print(f"Timelines: {', '.join(resolved_timelines)}")
    print(f"Test mode: {resolved_test_mode}")
    print(f"Cell filter: {resolved_cell_filter or 'all'}")
    print(f"Batch correction: {resolved_batch_correction}")
    if resolved_batch_correction:
        print(f"PCA kernel: {resolved_pca_kernel}")
        print(f"PCA components: {resolved_pca_n_components}")
    if manifest is not None:
        print(f"Split manifest: {split_manifest_path}")
        split_well_count = manifest[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
        print(f"Split subset: {subset} ({split_well_count} wells)")
    print(f"Output: {output_path}")
    print("=" * 60)

    count = 0
    correction_transform = None
    if resolved_batch_correction:
        selected_plates = experiment_df["Assay_Plate_Barcode"].drop_duplicates().tolist()
        correction_transform = fit_batch_correction(
            loader=loader,
            batch=resolved_batch,
            plates=selected_plates,
            kernel=resolved_pca_kernel,
            n_components=resolved_pca_n_components,
        )
        print(
            "Fitted batch correction on "
            f"{correction_transform.control_count} negcon wells with "
            f"{correction_transform.effective_n_components}/"
            f"{correction_transform.requested_n_components} components."
        )

    for cell in experiment_df.Cell_type.unique():
        cell_df = experiment_df.query("Cell_type==@cell")
        modality_1_perturbation = "compound"
        modality_1_experiments_df = cell_df.query("Perturbation==@modality_1_perturbation")

        for modality_1_timepoint in modality_1_experiments_df.Time.unique():
            count += 1
            modality_1_timepoint_df = modality_1_experiments_df.query("Time==@modality_1_timepoint")
            modality_1_df = load_profiles_for_plates(
                loader=loader,
                batch=resolved_batch,
                plates=list(modality_1_timepoint_df.Assay_Plate_Barcode.unique()),
                modality=modality_1_perturbation,
            )
            modality_1_df = filter_profiles_to_split_subset(
                modality_1_df,
                manifest,
                keep_negcon_on_selected_plates=True,
            )

            if modality_1_df.empty:
                print(
                    f"Skipping {modality_1_perturbation}_{cell}_{modality_1_timepoint}h - no data"
                )
                continue

            modality_1_df = apply_batch_correction(modality_1_df, correction_transform)
            modality_1_df[replicate_feature] = modality_1_df[replicate_feature].fillna("DMSO")
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
                continue
            _map_df, _fr = compute_map_and_fr(result, [replicate_feature], null_size)

            _fr_df = pd.DataFrame(
                {
                    "Description": [description],
                    "Modality": [modality_1_perturbation],
                    "Cell": [cell],
                    "time": [_time],
                    "timepoint": [modality_1_timepoint],
                    "fr": [_fr],
                }
            )
            replicability_fr_df = concat_profiles(replicability_fr_df, _fr_df)

            _map_df["Description"] = description
            _map_df["Modality"] = modality_1_perturbation
            _map_df["Cell"] = cell
            _map_df["time"] = _time
            _map_df["timepoint"] = modality_1_timepoint
            replicability_map_df = concat_profiles(replicability_map_df, _map_df)

            # --- Compound matching ---
            modality_1_df = remove_negcon_wells(modality_1_df)
            modality_1_consensus_df = compute_consensus(modality_1_df, replicate_feature)

            replicable_compounds = list(
                replicability_map_df[
                    (replicability_map_df.Description == description)
                    & replicability_map_df.above_q_threshold
                ][replicate_feature]
            )
            modality_1_consensus_df = filter_replicable(
                modality_1_consensus_df,
                replicable_compounds,
                id_col=replicate_feature,
            )

            modality_1_consensus_df = (
                modality_1_consensus_df.merge(
                    target1_metadata, on="Metadata_broad_sample", how="left"
                )
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
                _map_df, _fr = compute_map_and_fr(result, ["Metadata_matching_target"], null_size)

                _fr_df = pd.DataFrame(
                    {
                        "Description": [description],
                        "Modality": [modality_1_perturbation],
                        "Cell": [cell],
                        "time": [_time],
                        "timepoint": [modality_1_timepoint],
                        "fr": [_fr],
                    }
                )
                matching_fr_df = concat_profiles(matching_fr_df, _fr_df)

                _map_df["Description"] = description
                _map_df["Modality"] = modality_1_perturbation
                _map_df["Cell"] = cell
                _map_df["time"] = _time
                _map_df["timepoint"] = modality_1_timepoint
                matching_map_df = concat_profiles(matching_map_df, _map_df)

            # --- Process genetic perturbations (ORF/CRISPR) ---
            all_modality_2_experiments_df = cell_df.query("Perturbation!=@modality_1_perturbation")
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
                        batch=resolved_batch,
                        plates=list(modality_2_timepoint_df.Assay_Plate_Barcode.unique()),
                        modality=modality_2_perturbation,
                        attach_gene_target=True,
                    )
                    modality_2_df = filter_profiles_to_split_subset(
                        modality_2_df,
                        manifest,
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

                    if (
                        not replicability_map_df.empty
                        and replicability_map_df.Description.str.contains(description_2).any()
                    ):
                        pass  # Already computed
                    else:
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
                        _map_df, _fr = compute_map_and_fr(result, [replicate_feature], null_size)

                        _fr_df = pd.DataFrame(
                            {
                                "Description": [description_2],
                                "Modality": [modality_2_perturbation],
                                "Cell": [cell],
                                "time": [_time_2],
                                "timepoint": [modality_2_timepoint],
                                "fr": [_fr],
                            }
                        )
                        replicability_fr_df = concat_profiles(replicability_fr_df, _fr_df)

                        _map_df["Description"] = description_2
                        _map_df["Modality"] = modality_2_perturbation
                        _map_df["Cell"] = cell
                        _map_df["time"] = _time_2
                        _map_df["timepoint"] = modality_2_timepoint
                        replicability_map_df = concat_profiles(replicability_map_df, _map_df)

                    modality_2_df = remove_negcon_wells(modality_2_df)

                    modality_2_consensus_df = compute_consensus(
                        modality_2_df, "Metadata_broad_sample"
                    )

                    replicable_genes = list(
                        replicability_map_df[
                            (replicability_map_df.Description == description_2)
                            & replicability_map_df.above_q_threshold
                        ][replicate_feature]
                    )
                    modality_2_consensus_df = filter_replicable(
                        modality_2_consensus_df,
                        replicable_genes,
                        id_col=replicate_feature,
                    )

                    # Filter out reagents without a sister guide.
                    gene_counts = modality_2_consensus_df["Metadata_gene"].value_counts()
                    genes_without_sister = gene_counts[gene_counts == 1].index.to_list()

                    modality_2_consensus_for_matching_df = modality_2_consensus_df.loc[
                        ~modality_2_consensus_df["Metadata_gene"].isin(genes_without_sister)
                    ].reset_index(drop=True)

                    if modality_2_perturbation == "crispr":
                        if (
                            matching_map_df.empty
                            or not matching_map_df.Description.str.contains(description_2).any()
                        ):
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

                            _map_df, _fr = compute_map_and_fr(
                                result, ["Metadata_matching_target"], null_size
                            )

                            _fr_df = pd.DataFrame(
                                {
                                    "Description": [description_2],
                                    "Modality": [modality_2_perturbation],
                                    "Cell": [cell],
                                    "time": [_time_2],
                                    "timepoint": [modality_2_timepoint],
                                    "fr": [_fr],
                                }
                            )
                            matching_fr_df = concat_profiles(matching_fr_df, _fr_df)

                            _map_df["Description"] = description_2
                            _map_df["Modality"] = modality_2_perturbation
                            _map_df["Cell"] = cell
                            _map_df["time"] = _time_2
                            _map_df["timepoint"] = modality_2_timepoint
                            matching_map_df = concat_profiles(matching_map_df, _map_df)

                    perturbed_genes = list(set(modality_2_consensus_df.Metadata_matching_target))
                    modality_1_targets_df = (
                        modality_1_consensus_df[
                            ["Metadata_broad_sample", "Metadata_matching_target"]
                        ]
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
                        continue

                    description_cross = (
                        f"{modality_1_perturbation}_{cell}_{_time}"
                        f"-{modality_2_perturbation}_{cell}_{_time_2}"
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
                        continue

                    _map_df, _fr = compute_map_and_fr(
                        result, ["Metadata_matching_target"], null_size
                    )

                    _time_1 = _time
                    _fr_df = pd.DataFrame(
                        {
                            "Description": [description_cross],
                            "Modality1": [modality_1_perturbation],
                            "Modality2": [modality_2_perturbation],
                            "Cell": [cell],
                            "time1": [_time_1],
                            "time2": [_time_2],
                            "timepoint1": [modality_1_timepoint],
                            "timepoint2": [modality_2_timepoint],
                            "fr": [_fr],
                        }
                    )
                    gene_compound_matching_fr_df = concat_profiles(
                        gene_compound_matching_fr_df, _fr_df
                    )

                    _map_df["Description"] = description_cross
                    _map_df["Modality1"] = modality_1_perturbation
                    _map_df["Modality2"] = modality_2_perturbation
                    _map_df["Cell"] = cell
                    _map_df["time1"] = _time_1
                    _map_df["time2"] = _time_2
                    _map_df["timepoint1"] = modality_1_timepoint
                    _map_df["timepoint2"] = modality_2_timepoint
                    gene_compound_matching_map_df = concat_profiles(
                        gene_compound_matching_map_df, _map_df
                    )

    # --- Save results ---
    print("\nSaving results...")

    replicability_map_df.to_csv(output_path / "cellprofiler_replicability_map.csv", index=False)
    replicability_fr_df.to_csv(output_path / "cellprofiler_replicability_fr.csv", index=False)
    matching_map_df.to_csv(output_path / "cellprofiler_matching_map.csv", index=False)
    matching_fr_df.to_csv(output_path / "cellprofiler_matching_fr.csv", index=False)
    gene_compound_matching_map_df.to_csv(
        output_path / "cellprofiler_gene_compound_matching_map.csv", index=False
    )
    gene_compound_matching_fr_df.to_csv(
        output_path / "cellprofiler_gene_compound_matching_fr.csv", index=False
    )

    # Summary tables
    if not replicability_fr_df.empty:
        pivot_repl = replicability_fr_df.pivot_table(
            values="fr", index=["Modality", "time"], columns="Cell", aggfunc="first"
        )
        pivot_repl.to_csv(tables_dir / "replicability_summary.csv")

    if not matching_fr_df.empty:
        pivot_match = matching_fr_df.pivot_table(
            values="fr", index=["Modality", "time"], columns="Cell", aggfunc="first"
        )
        pivot_match.to_csv(tables_dir / "matching_summary.csv")

    if not gene_compound_matching_fr_df.empty:
        pivot_cross = gene_compound_matching_fr_df.pivot_table(
            values="fr", index=["Modality1", "Modality2"], columns="Cell", aggfunc="first"
        )
        pivot_cross.to_csv(tables_dir / "gene_compound_matching_summary.csv")

    # --- Generate figures ---
    print("\nGenerating figures...")

    if not replicability_fr_df.empty:
        plot_replicability_barplot(
            replicability_fr_df, figures_dir / "replicability_fr_barplot.png"
        )
        plot_replicability_fr_faceted(
            replicability_fr_df, figures_dir / "replicability_fr_faceted.png"
        )

    if not replicability_map_df.empty:
        plot_replicability_map_boxplot(
            replicability_map_df, figures_dir / "replicability_map_boxplot.png"
        )

    if not matching_fr_df.empty:
        plot_matching_barplot(matching_fr_df, figures_dir / "matching_fr_barplot.png")
        plot_matching_fr_faceted(matching_fr_df, figures_dir / "matching_fr_faceted.png")

    if not matching_map_df.empty:
        plot_matching_map_boxplot(matching_map_df, figures_dir / "matching_map_boxplot.png")

    if not gene_compound_matching_fr_df.empty:
        plot_cross_modality_barplot(
            gene_compound_matching_fr_df, figures_dir / "gene_compound_matching_fr_barplot.png"
        )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if not replicability_fr_df.empty:
        print("\nReplicability (Fraction Retrieved):")
        print(replicability_fr_df[["Description", "fr"]].to_string(index=False))

    if not matching_fr_df.empty:
        print("\nTarget Matching (Fraction Retrieved):")
        print(matching_fr_df[["Description", "fr"]].to_string(index=False))

    if not gene_compound_matching_fr_df.empty:
        print("\nGene-Compound Matching (Fraction Retrieved):")
        print(gene_compound_matching_fr_df[["Description", "fr"]].to_string(index=False))

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print("=" * 60)
