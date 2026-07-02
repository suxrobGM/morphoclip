"""Stable benchmark evaluation aligned to the 2024 Chandrasekaran reference pipeline.

This module is the orchestration coordinator for the CPJUMP1 "stable copairs"
benchmark. Config resolution lives in :mod:`benchmark.stable_config`, per-modality
evaluation in :mod:`benchmark.stable_compound` and :mod:`benchmark.stable_genetic`,
and result accumulation / persistence in :mod:`benchmark.stable_results`. The CLI
wrapper (``morphoclip benchmark``) is a thin adapter over :func:`run_stable_benchmark`.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

from benchmark.data import (
    ProfileLoader,
    filter_experiment_metadata_to_split_subset,
    get_timepoint_label,
)
from benchmark.stable_compound import evaluate_compound
from benchmark.stable_config import CONFIG_PATH, BenchmarkParams, resolve_params
from benchmark.stable_genetic import evaluate_genetic
from benchmark.stable_helpers import fit_batch_correction
from benchmark.stable_results import StableResults

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _print_run_header(params: BenchmarkParams) -> None:
    """Print the run configuration banner."""
    print("=" * 60)
    print("CPJUMP1 Benchmark Evaluation (Stable Mode)")
    print("=" * 60)
    print(f"Profiles: {params.profiles_path}")
    print(f"Batch: {params.batch}")
    print(f"Timelines: {', '.join(params.timelines)}")
    print(f"Test mode: {params.test_mode}")
    print(f"Cell filter: {params.cell_filter or 'all'}")
    print(f"Batch correction: {params.batch_correction}")
    if params.batch_correction:
        print(f"PCA kernel: {params.pca_kernel}")
        print(f"PCA components: {params.pca_n_components}")
    if params.manifest is not None:
        print(f"Split manifest: {params.split_manifest_path}")
        split_well_count = (
            params.manifest[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
        )
        print(f"Split subset: {params.subset} ({split_well_count} wells)")
    print(f"Output: {params.output_path}")
    print("=" * 60)


def _load_experiment_metadata(params: BenchmarkParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter experiment metadata + target annotations.

    Exits the process (``sys.exit(1)``) when required input files are missing or
    no experiments survive the timeline/cell filters.
    """
    exp_meta_path = params.project_root / "output/benchmark/input/experiment-metadata.tsv"
    target_ann_path = (
        params.project_root
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
        .query(f"Batch=='{params.batch}'")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )
    experiment_df = experiment_df.drop(
        experiment_df[
            (experiment_df.Perturbation == "compound") & (experiment_df.Cell_line == "Cas9")
        ].index
    )
    experiment_df = filter_experiment_metadata_to_split_subset(experiment_df, params.manifest)
    experiment_df["timeline"] = experiment_df.apply(
        lambda row: get_timepoint_label(row["Perturbation"], row["Time"]),
        axis=1,
    )
    experiment_df = experiment_df[experiment_df["timeline"].isin(params.timelines)].reset_index(
        drop=True
    )

    if experiment_df.empty:
        print(
            "ERROR: No experiments matched the selected timeline(s): "
            f"{', '.join(params.timelines)} for batch {params.batch}"
        )
        sys.exit(1)

    target1_metadata = pd.read_csv(
        target_ann_path, sep="\t", usecols=["broad_sample", "target_list"]
    ).rename(
        columns={"broad_sample": "Metadata_broad_sample", "target_list": "Metadata_target_list"}
    )

    if params.test_mode:
        print("TEST MODE: Using limited data")
        experiment_df = experiment_df.head(8)
        print(experiment_df)

    if params.cell_filter:
        experiment_df = experiment_df.query(f"Cell_type=='{params.cell_filter}'")

    if experiment_df.empty:
        print("ERROR: No experiments remained after applying filters.")
        sys.exit(1)

    return experiment_df, target1_metadata


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
    params = resolve_params(
        config=config,
        profiles_dir=profiles_dir,
        output_dir=output_dir,
        batch=batch,
        test_mode=test_mode,
        cell_filter=cell_filter,
        batch_correction=batch_correction,
        pca_kernel=pca_kernel,
        pca_n_components=pca_n_components,
        split_manifest=split_manifest,
        split_subset=split_subset,
        timelines=timelines,
    )

    params.output_path.mkdir(parents=True, exist_ok=True)
    params.figures_dir.mkdir(parents=True, exist_ok=True)
    params.tables_dir.mkdir(parents=True, exist_ok=True)

    experiment_df, target1_metadata = _load_experiment_metadata(params)

    batch_size = 100000 if not params.test_mode else 20000
    null_size = 100000 if not params.test_mode else 10000
    loader = ProfileLoader(params.profiles_path)
    results = StableResults()

    _print_run_header(params)

    correction_transform = None
    if params.batch_correction:
        selected_plates = experiment_df["Assay_Plate_Barcode"].drop_duplicates().tolist()
        correction_transform = fit_batch_correction(
            loader=loader,
            batch=params.batch,
            plates=selected_plates,
            kernel=params.pca_kernel,
            n_components=params.pca_n_components,
        )
        print(
            "Fitted batch correction on "
            f"{correction_transform.control_count} negcon wells with "
            f"{correction_transform.effective_n_components}/"
            f"{correction_transform.requested_n_components} components."
        )

    count = 0
    for cell in experiment_df.Cell_type.unique():
        cell_df = experiment_df.query("Cell_type==@cell")
        modality_1_experiments_df = cell_df.query("Perturbation == 'compound'")

        for modality_1_timepoint in modality_1_experiments_df.Time.unique():
            count += 1
            compound = evaluate_compound(
                results,
                loader=loader,
                params=params,
                correction_transform=correction_transform,
                target1_metadata=target1_metadata,
                cell=cell,
                modality_1_experiments_df=modality_1_experiments_df,
                modality_1_timepoint=modality_1_timepoint,
                count=count,
                batch_size=batch_size,
                null_size=null_size,
            )
            if compound is None:
                continue
            evaluate_genetic(
                results,
                loader=loader,
                params=params,
                correction_transform=correction_transform,
                cell=cell,
                cell_df=cell_df,
                compound=compound,
                count=count,
                batch_size=batch_size,
                null_size=null_size,
            )

    results.save(params.output_path, params.tables_dir)
    results.generate_figures(params.figures_dir)
    results.print_summary(params.output_path)
