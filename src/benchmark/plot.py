"""Comparison tables and plots for benchmark fraction-retrieved outputs."""

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class TaskSpec:
    """Metadata describing one benchmark comparison table."""

    name: str
    filename: str
    key_columns: tuple[str, ...]
    title: str


@dataclass(frozen=True)
class RunSpec:
    """One named benchmark output directory."""

    label: str
    run_dir: Path

    @property
    def slug(self) -> str:
        """Filesystem- and column-safe label slug."""
        normalized = re.sub(r"[^a-z0-9]+", "_", self.label.strip().lower()).strip("_")
        return normalized or "run"


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        name="replicability",
        filename="cellprofiler_replicability_fr.csv",
        key_columns=("Description", "Modality", "Cell", "time", "timepoint"),
        title="Replicability",
    ),
    TaskSpec(
        name="matching",
        filename="cellprofiler_matching_fr.csv",
        key_columns=("Description", "Modality", "Cell", "time", "timepoint"),
        title="Target Matching",
    ),
    TaskSpec(
        name="gene_compound_matching",
        filename="cellprofiler_gene_compound_matching_fr.csv",
        key_columns=(
            "Description",
            "Modality1",
            "Modality2",
            "Cell",
            "time1",
            "time2",
            "timepoint1",
            "timepoint2",
        ),
        title="Gene-Compound Matching",
    ),
)


def _task_specs_by_name() -> dict[str, TaskSpec]:
    return {spec.name: spec for spec in TASK_SPECS}


def _build_plot_label(row: pd.Series, spec: TaskSpec) -> str:
    if spec.name in {"replicability", "matching"}:
        return f"{row['Modality']} | {row['Cell']} | {row['time']}"
    return (
        f"{row['Modality1']}->{row['Modality2']} | {row['Cell']} | {row['time1']}->{row['time2']}"
    )


def load_fraction_retrieved(run_dir: str | Path, spec: TaskSpec) -> pd.DataFrame:
    """Load one fraction-retrieved CSV from a benchmark output directory."""
    path = Path(run_dir) / spec.filename
    if not path.exists():
        return pd.DataFrame(columns=[*spec.key_columns, "fr"])

    df = pd.read_csv(path)
    missing = set(spec.key_columns).difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing required columns: {missing_list}")
    if "fr" not in df.columns:
        raise ValueError(f"{path} is missing required column: fr")
    return df.loc[:, [*spec.key_columns, "fr"]].copy()


def normalize_run_specs(
    *,
    run_specs: list[RunSpec] | None = None,
    baseline_dir: str | Path | None = None,
    candidate_dir: str | Path | None = None,
    baseline_label: str = "Baseline",
    candidate_label: str = "CellCLIP",
) -> list[RunSpec]:
    """Normalize comparison inputs to a list of named runs."""
    if run_specs:
        return [RunSpec(label=spec.label, run_dir=Path(spec.run_dir)) for spec in run_specs]

    if baseline_dir is None or candidate_dir is None:
        raise ValueError("Either run_specs or baseline_dir/candidate_dir must be provided")

    return [
        RunSpec(label=baseline_label, run_dir=Path(baseline_dir)),
        RunSpec(label=candidate_label, run_dir=Path(candidate_dir)),
    ]


def collect_fraction_retrieved(
    run_specs: list[RunSpec],
    task_name: str,
) -> pd.DataFrame:
    """Collect one task's FR tables across all named benchmark runs."""
    spec = _task_specs_by_name()[task_name]
    frames: list[pd.DataFrame] = []
    for run in run_specs:
        df = load_fraction_retrieved(run.run_dir, spec)
        if df.empty:
            continue
        df = df.copy()
        df["profile"] = run.label
        df["profile_slug"] = run.slug
        df["task"] = spec.name
        df["task_title"] = spec.title
        df["plot_label"] = df.apply(_build_plot_label, axis=1, spec=spec)
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                *spec.key_columns,
                "fr",
                "profile",
                "profile_slug",
                "task",
                "task_title",
                "plot_label",
            ]
        )

    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .sort_values(
            [*spec.key_columns, "profile"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def collect_all_fraction_retrieved(
    run_specs: list[RunSpec],
) -> dict[str, pd.DataFrame]:
    """Collect all benchmark FR outputs for all named runs."""
    return {spec.name: collect_fraction_retrieved(run_specs, spec.name) for spec in TASK_SPECS}


def build_wide_comparison(
    df: pd.DataFrame,
    task_name: str,
) -> pd.DataFrame:
    """Pivot one long comparison table into a wide per-profile FR table."""
    spec = _task_specs_by_name()[task_name]
    if df.empty:
        return pd.DataFrame(columns=[*spec.key_columns, "task", "task_title", "plot_label"])

    key_columns = list(spec.key_columns) + ["task", "task_title", "plot_label"]
    wide = df.pivot_table(
        index=key_columns,
        columns="profile_slug",
        values="fr",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    rename_map = {
        run_slug: f"{run_slug}_fr" for run_slug in wide.columns if run_slug not in key_columns
    }
    return wide.rename(columns=rename_map).sort_values(list(spec.key_columns), kind="stable")


def build_overall_summary(
    comparisons: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute per-task and overall mean FR summaries for each profile."""
    frames: list[pd.DataFrame] = []
    for df in comparisons.values():
        if df.empty:
            continue
        task_summary = (
            df.groupby(["profile", "profile_slug", "task", "task_title"], dropna=False)["fr"]
            .agg(mean_fr="mean", min_fr="min", max_fr="max", n="count")
            .reset_index()
        )
        frames.append(task_summary)

    if not frames:
        return pd.DataFrame(
            columns=[
                "profile",
                "profile_slug",
                "task",
                "task_title",
                "mean_fr",
                "min_fr",
                "max_fr",
                "n",
                "rank",
            ]
        )

    summary = pd.concat(frames, ignore_index=True, sort=False)
    overall = (
        summary.groupby(["profile", "profile_slug"], dropna=False)
        .agg(
            mean_fr=("mean_fr", "mean"),
            min_fr=("min_fr", "min"),
            max_fr=("max_fr", "max"),
            n=("n", "sum"),
        )
        .reset_index()
    )
    overall["task"] = "overall"
    overall["task_title"] = "Overall"

    summary = pd.concat([summary, overall], ignore_index=True, sort=False)
    summary["rank"] = (
        summary.groupby("task", dropna=False)["mean_fr"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return summary.sort_values(["task", "rank", "profile"], kind="stable").reset_index(drop=True)


def save_comparison_tables(
    comparisons: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save per-task, wide, combined, and overall comparison tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    combined_frames: list[pd.DataFrame] = []
    for task_name, df in comparisons.items():
        path = output_dir / f"{task_name}_comparison.csv"
        df.to_csv(path, index=False)
        saved_paths[task_name] = path

        wide = build_wide_comparison(df, task_name)
        wide_path = output_dir / f"{task_name}_comparison_wide.csv"
        wide.to_csv(wide_path, index=False)
        saved_paths[f"{task_name}_wide"] = wide_path
        combined_frames.append(df)

    combined_path = output_dir / "benchmark_comparison.csv"
    if combined_frames:
        pd.concat(combined_frames, ignore_index=True, sort=False).to_csv(combined_path, index=False)
    else:
        pd.DataFrame().to_csv(combined_path, index=False)
    saved_paths["combined"] = combined_path

    overall = build_overall_summary(comparisons)
    overall_path = output_dir / "overall_comparison.csv"
    overall.to_csv(overall_path, index=False)
    saved_paths["overall"] = overall_path

    return saved_paths


def _plot_task_comparison(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Render a multi-profile horizontal FR comparison for one benchmark task."""
    if df.empty:
        return

    plot_df = df.copy()
    order = (
        plot_df.groupby("plot_label", dropna=False)["fr"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_df["plot_label"] = pd.Categorical(plot_df["plot_label"], categories=order, ordered=True)

    sns.set_theme(style="whitegrid")
    fig_height = max(4.0, 0.55 * len(order) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    sns.scatterplot(
        data=plot_df,
        x="fr",
        y="plot_label",
        hue="profile",
        hue_order=sorted(plot_df["profile"].dropna().unique()),
        s=90,
        ax=ax,
    )

    max_fr = float(plot_df["fr"].max()) if not plot_df["fr"].empty else 0.0
    ax.set_xlim(0, max(1.05, max_fr + 0.1))
    ax.set_xlabel("Fraction Retrieved")
    ax.set_ylabel("")
    ax.set_title(plot_df["task_title"].iat[0])
    ax.grid(axis="x", color="#d9dde1", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.legend(title="Profile", loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overall_comparison(
    overall_summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Render the per-task and overall mean FR summary for all profiles."""
    if overall_summary.empty:
        return

    plot_df = overall_summary.copy()
    task_order = [spec.name for spec in TASK_SPECS] + ["overall"]
    title_map = {spec.name: spec.title for spec in TASK_SPECS}
    title_map["overall"] = "Overall"
    plot_df = plot_df[plot_df["task"].isin(task_order)].copy()
    plot_df["task_title"] = plot_df["task"].map(title_map)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        x="task_title",
        y="mean_fr",
        hue="profile",
        order=[title_map[name] for name in task_order if name in plot_df["task"].values],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean Fraction Retrieved")
    ax.set_title("Overall Benchmark Comparison")
    ax.set_ylim(0, max(1.05, float(plot_df["mean_fr"].max()) + 0.1))
    ax.legend(title="Profile", loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_comparison(
    comparisons: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Render per-task comparison plots plus an overall summary plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for spec in TASK_SPECS:
        df = comparisons[spec.name]
        if df.empty:
            continue
        path = output_dir / f"{spec.name}_comparison.png"
        _plot_task_comparison(df, path)
        saved_paths[spec.name] = path

    overall = build_overall_summary(comparisons)
    if not overall.empty:
        overall_path = output_dir / "overall_comparison.png"
        plot_overall_comparison(overall, overall_path)
        saved_paths["overall"] = overall_path

    return saved_paths


def generate_benchmark_comparison(
    *,
    run_specs: list[RunSpec] | None = None,
    baseline_dir: str | Path | None = None,
    candidate_dir: str | Path | None = None,
    output_dir: str | Path,
    baseline_label: str = "Baseline",
    candidate_label: str = "CellCLIP",
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, Path]]:
    """Generate merged comparison tables and plots for two or more benchmark runs."""
    normalized_runs = normalize_run_specs(
        run_specs=run_specs,
        baseline_dir=baseline_dir,
        candidate_dir=candidate_dir,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
    )
    if len(normalized_runs) < 2:
        raise ValueError("At least two benchmark runs are required for comparison")

    comparisons = collect_all_fraction_retrieved(normalized_runs)
    table_paths = save_comparison_tables(comparisons, output_dir)
    plot_paths = plot_benchmark_comparison(
        comparisons=comparisons,
        output_dir=output_dir,
    )
    return comparisons, table_paths, plot_paths
