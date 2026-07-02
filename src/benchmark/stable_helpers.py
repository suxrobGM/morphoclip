"""Helper utilities for the stable benchmark script."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

from benchmark.data import ProfileLoader, get_feature_columns
from benchmark.metrics import compute_fraction_retrieved, compute_map
from benchmark.profile_ops import concat_profiles

__all__ = [
    "concat_profiles",
    "load_profiles_for_plates",
    "BatchCorrectionTransform",
    "fit_batch_correction",
    "apply_batch_correction",
    "run_with_unpaired_guard",
    "compute_map_and_fr",
    "plot_replicability_barplot",
    "plot_matching_barplot",
    "plot_cross_modality_barplot",
    "plot_replicability_map_boxplot",
    "plot_matching_map_boxplot",
    "plot_replicability_fr_faceted",
    "plot_matching_fr_faceted",
]


def load_profiles_for_plates(
    loader: ProfileLoader,
    batch: str,
    plates: list[str],
    modality: str,
    attach_gene_target: bool = False,
) -> pd.DataFrame:
    """Load plate profiles for a modality and optionally assign gene targets."""
    dfs = []
    for plate in plates:
        try:
            df = loader.load_plate(batch, plate).assign(Metadata_modality=modality)
            if attach_gene_target:
                df = df.assign(Metadata_matching_target=lambda x: x.Metadata_gene)
            dfs.append(df)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True, join="inner")


@dataclass(frozen=True)
class BatchCorrectionTransform:
    """Fitted eval-time batch correction state."""

    feature_cols: tuple[str, ...]
    output_feature_cols: tuple[str, ...]
    kernel_pca: KernelPCA
    scaler: StandardScaler
    requested_n_components: int
    effective_n_components: int
    control_count: int


def _coerce_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Return a float32 feature matrix for the provided columns."""
    return df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)


def fit_batch_correction(
    loader: ProfileLoader,
    batch: str,
    plates: list[str],
    *,
    kernel: str = "linear",
    n_components: int = 500,
    control_col: str = "Metadata_control_type",
    control_value: str = "negcon",
) -> BatchCorrectionTransform:
    """Fit a pooled control-based KernelPCA + StandardScaler transform."""
    control_frames: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None

    for plate in plates:
        try:
            df = loader.load_plate(batch, plate)
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
            continue
        if feature_cols is None:
            feature_cols = get_feature_columns(df)
            if not feature_cols:
                raise ValueError("Batch correction requires at least one feature column.")

        controls = df[df[control_col] == control_value]
        if not controls.empty:
            control_frames.append(controls)

    if not control_frames:
        raise ValueError(
            f"Batch correction requested, but no {control_value!r} wells were found "
            f"across {len(plates)} selected plates."
        )

    assert feature_cols is not None
    control_df = pd.concat(control_frames, ignore_index=True, join="inner")
    control_values = _coerce_feature_matrix(control_df, feature_cols)
    control_count = int(control_values.shape[0])
    if control_count < 2:
        raise ValueError("Batch correction requires at least 2 control wells to fit KernelPCA.")

    effective_n_components = min(
        int(n_components), control_values.shape[0], control_values.shape[1]
    )
    if effective_n_components < 1:
        raise ValueError("Batch correction resolved to zero PCA components.")

    kernel_pca = KernelPCA(n_components=effective_n_components, kernel=kernel)
    control_projected = kernel_pca.fit_transform(control_values)
    scaler = StandardScaler()
    scaler.fit(control_projected)

    return BatchCorrectionTransform(
        feature_cols=tuple(feature_cols),
        output_feature_cols=tuple(
            f"corrected_feature_{index:04d}" for index in range(effective_n_components)
        ),
        kernel_pca=kernel_pca,
        scaler=scaler,
        requested_n_components=int(n_components),
        effective_n_components=effective_n_components,
        control_count=control_count,
    )


def apply_batch_correction(
    df: pd.DataFrame,
    transform: BatchCorrectionTransform | None,
) -> pd.DataFrame:
    """Apply a fitted batch-correction transform to a profile dataframe."""
    if transform is None or df.empty:
        return df

    missing = [col for col in transform.feature_cols if col not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Profile data is missing batch-correction feature columns: {preview}")

    corrected = df.copy()
    feature_values = _coerce_feature_matrix(corrected, list(transform.feature_cols))
    projected = transform.kernel_pca.transform(feature_values)
    corrected_values = transform.scaler.transform(projected)

    metadata_cols = [col for col in corrected.columns if col.startswith("Metadata")]
    metadata_df = corrected.loc[:, metadata_cols].copy()
    corrected_df = pd.DataFrame(
        corrected_values,
        columns=list(transform.output_feature_cols),
        index=corrected.index,
    )
    return pd.concat([metadata_df, corrected_df], axis=1)


def run_with_unpaired_guard(
    func: Callable[..., pd.DataFrame], *args: Any, **kwargs: Any
) -> pd.DataFrame:
    """Run an evaluation function and suppress copairs UnpairedException."""
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - depends on runtime copairs behavior
        if "old copairs API" in str(exc):
            raise
        if type(exc).__name__ != "UnpairedException" and str(exc) != "dict_pairs empty":
            raise
        print(f"Warning: skipping run with no valid pairs ({exc}).")
        return pd.DataFrame()


def compute_map_and_fr(
    result: pd.DataFrame,
    group_cols: list[str],
    null_size: int,
    threshold: float = 0.05,
) -> tuple[pd.DataFrame, float]:
    """Compute mAP table and fraction retrieved in stable copairs mode."""
    map_df = compute_map(
        result=result,
        group_cols=group_cols,
        threshold=threshold,
        null_size=null_size,
        copairs_mode="stable",
    )
    if len(map_df) == 0:
        return map_df, 0.0
    return map_df, compute_fraction_retrieved(map_df)


def plot_replicability_barplot(fr_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 6))
    fr_df = fr_df.copy()
    fr_df["label"] = fr_df["Modality"] + "_" + fr_df["time"]
    sns.barplot(data=fr_df, x="label", y="fr", hue="Cell", ax=ax, palette="Set2")
    ax.set_xlabel("Modality / Timepoint")
    ax.set_ylabel("Fraction Retrieved")
    ax.set_title("Replicability: Fraction Retrieved (q < 0.05)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_matching_barplot(fr_df: pd.DataFrame, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    fr_df = fr_df.copy()
    fr_df["label"] = fr_df["Modality"] + "_" + fr_df["time"]
    sns.barplot(data=fr_df, x="label", y="fr", hue="Cell", ax=ax, palette="Set3")
    ax.set_xlabel("Modality / Timepoint")
    ax.set_ylabel("Fraction Retrieved")
    ax.set_title("Target Matching: Fraction Retrieved (q < 0.05)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_cross_modality_barplot(fr_df: pd.DataFrame, output_path: Path):
    if fr_df.empty:
        print("No cross-modality results to plot")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    fr_df = fr_df.copy()
    fr_df["label"] = fr_df["Modality1"] + " vs " + fr_df["Modality2"]
    sns.barplot(data=fr_df, x="label", y="fr", hue="Cell", ax=ax, palette="Set1")
    ax.set_xlabel("Modality Comparison")
    ax.set_ylabel("Fraction Retrieved")
    ax.set_title("Cross-Modality Matching: Fraction Retrieved (q < 0.05)")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_replicability_map_boxplot(map_df: pd.DataFrame, output_path: Path):
    """Box plot of mAP values faceted by Cell (row) and time (col)."""
    if map_df.empty:
        print("No replicability mAP data to plot")
        return

    cells = map_df["Cell"].unique()
    times = map_df["time"].unique()
    n_rows, n_cols = len(cells), len(times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, cell in enumerate(cells):
        for j, time in enumerate(times):
            ax = axes[i, j]
            subset = map_df[(map_df["Cell"] == cell) & (map_df["time"] == time)]
            if not subset.empty:
                sns.boxplot(
                    data=subset,
                    x="Modality",
                    y="mean_average_precision",
                    ax=ax,
                    palette="Set2",
                )
                ax.set_ylim(0, 1)
            ax.set_title(f"{cell} / {time}")
            ax.set_xlabel("Perturbation" if i == n_rows - 1 else "")
            ax.set_ylabel("mAP" if j == 0 else "")

    plt.suptitle("Replicability: mAP Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_matching_map_boxplot(map_df: pd.DataFrame, output_path: Path):
    """Box plot of matching mAP values faceted by Cell (row) and time (col)."""
    if map_df.empty:
        print("No matching mAP data to plot")
        return

    cells = map_df["Cell"].unique()
    times = map_df["time"].unique()
    n_rows, n_cols = len(cells), len(times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, cell in enumerate(cells):
        for j, time in enumerate(times):
            ax = axes[i, j]
            subset = map_df[(map_df["Cell"] == cell) & (map_df["time"] == time)]
            if not subset.empty:
                sns.boxplot(
                    data=subset,
                    x="Modality",
                    y="mean_average_precision",
                    ax=ax,
                    palette="Set3",
                )
                ax.set_ylim(0, 1)
            ax.set_title(f"{cell} / {time}")
            ax.set_xlabel("Perturbation" if i == n_rows - 1 else "")
            ax.set_ylabel("mAP" if j == 0 else "")

    plt.suptitle("Target Matching: mAP Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_replicability_fr_faceted(fr_df: pd.DataFrame, output_path: Path):
    """Bar plot of FR faceted by Cell (row) and time (col)."""
    if fr_df.empty:
        print("No replicability FR data to plot")
        return

    cells = fr_df["Cell"].unique()
    times = fr_df["time"].unique()
    n_rows, n_cols = len(cells), len(times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, cell in enumerate(cells):
        for j, time in enumerate(times):
            ax = axes[i, j]
            subset = fr_df[(fr_df["Cell"] == cell) & (fr_df["time"] == time)]
            if not subset.empty:
                sns.barplot(data=subset, x="Modality", y="fr", ax=ax, palette="Set2")
                ax.set_ylim(0, 1)
            ax.set_title(f"{cell} / {time}")
            ax.set_xlabel("Perturbation" if i == n_rows - 1 else "")
            ax.set_ylabel("Fraction Retrieved" if j == 0 else "")

    plt.suptitle("Replicability: Fraction Retrieved", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_matching_fr_faceted(fr_df: pd.DataFrame, output_path: Path):
    """Bar plot of matching FR faceted by Cell (row) and time (col)."""
    if fr_df.empty:
        print("No matching FR data to plot")
        return

    cells = fr_df["Cell"].unique()
    times = fr_df["time"].unique()
    n_rows, n_cols = len(cells), len(times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, cell in enumerate(cells):
        for j, time in enumerate(times):
            ax = axes[i, j]
            subset = fr_df[(fr_df["Cell"] == cell) & (fr_df["time"] == time)]
            if not subset.empty:
                sns.barplot(data=subset, x="Modality", y="fr", ax=ax, palette="Set3")
                ax.set_ylim(0, 1)
            ax.set_title(f"{cell} / {time}")
            ax.set_xlabel("Perturbation" if i == n_rows - 1 else "")
            ax.set_ylabel("Fraction Retrieved" if j == 0 else "")

    plt.suptitle("Target Matching: Fraction Retrieved", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
