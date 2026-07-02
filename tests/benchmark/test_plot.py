"""Tests for benchmark comparison table generation."""

from pathlib import Path

import pandas as pd

from benchmark.plot import RunSpec, generate_benchmark_comparison


def _write_run_dir(root: Path, *, label: str, replicability_fr: float, matching_fr: float) -> Path:
    run_dir = root / label
    run_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "Description": ["compound_A549_long"],
            "Modality": ["compound"],
            "Cell": ["A549"],
            "time": ["long"],
            "timepoint": [48],
            "fr": [replicability_fr],
        }
    ).to_csv(run_dir / "cellprofiler_replicability_fr.csv", index=False)

    pd.DataFrame(
        {
            "Description": ["compound_A549_long"],
            "Modality": ["compound"],
            "Cell": ["A549"],
            "time": ["long"],
            "timepoint": [48],
            "fr": [matching_fr],
        }
    ).to_csv(run_dir / "cellprofiler_matching_fr.csv", index=False)

    pd.DataFrame(
        {
            "Description": ["compound_A549_long-crispr_A549_long"],
            "Modality1": ["compound"],
            "Modality2": ["crispr"],
            "Cell": ["A549"],
            "time1": ["long"],
            "time2": ["long"],
            "timepoint1": [48],
            "timepoint2": [144],
            "fr": [0.0],
        }
    ).to_csv(run_dir / "cellprofiler_gene_compound_matching_fr.csv", index=False)

    return run_dir


def test_generate_benchmark_comparison_supports_multiple_profiles(tmp_path: Path) -> None:
    baseline_dir = _write_run_dir(
        tmp_path,
        label="baseline",
        replicability_fr=0.80,
        matching_fr=0.20,
    )
    cellclip_dir = _write_run_dir(
        tmp_path,
        label="cellclip",
        replicability_fr=0.70,
        matching_fr=0.25,
    )
    current_dir = _write_run_dir(
        tmp_path,
        label="current",
        replicability_fr=0.90,
        matching_fr=0.30,
    )

    comparisons, table_paths, plot_paths = generate_benchmark_comparison(
        run_specs=[
            RunSpec(label="Baseline", run_dir=baseline_dir),
            RunSpec(label="CellCLIP", run_dir=cellclip_dir),
            RunSpec(label="Current", run_dir=current_dir),
        ],
        output_dir=tmp_path / "comparison",
    )

    replicability = comparisons["replicability"]
    assert set(replicability["profile"]) == {"Baseline", "CellCLIP", "Current"}

    overall = pd.read_csv(table_paths["overall"])
    overall_only = overall[overall["task"] == "overall"]
    assert set(overall_only["profile"]) == {"Baseline", "CellCLIP", "Current"}
    assert overall_only.sort_values("rank")["profile"].tolist()[0] == "Current"

    wide = pd.read_csv(table_paths["replicability_wide"])
    assert {"baseline_fr", "cellclip_fr", "current_fr"} <= set(wide.columns)

    assert "overall" in plot_paths
