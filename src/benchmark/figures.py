"""The seven per-run figures the stable benchmark writes.

Split out of stable_helpers.py, which mixed profile loading, batch correction
and plotting in one module.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
