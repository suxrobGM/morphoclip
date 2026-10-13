"""The seven per-run figures the stable benchmark writes."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _labelled_barplot(
    fr_df: pd.DataFrame,
    output_path: Path,
    *,
    label: pd.Series,
    figsize: tuple[int, int],
    palette: str,
    xlabel: str,
    title: str,
) -> None:
    """Bar plot of FR per *label*, hued by Cell."""
    fig, ax = plt.subplots(figsize=figsize)
    fr_df = fr_df.assign(label=label)
    sns.barplot(data=fr_df, x="label", y="fr", hue="Cell", ax=ax, palette=palette)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction Retrieved")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _faceted(
    df: pd.DataFrame,
    output_path: Path,
    *,
    plot: Callable[..., object],
    y: str,
    ylabel: str,
    palette: str,
    title: str,
) -> None:
    """Grid of per-Modality plots faceted by Cell (row) and time (col)."""
    cells = df["Cell"].unique()
    times = df["time"].unique()
    n_rows, n_cols = len(cells), len(times)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for i, cell in enumerate(cells):
        for j, time in enumerate(times):
            ax = axes[i, j]
            subset = df[(df["Cell"] == cell) & (df["time"] == time)]
            if not subset.empty:
                plot(data=subset, x="Modality", y=y, ax=ax, palette=palette)
                ax.set_ylim(0, 1)
            ax.set_title(f"{cell} / {time}")
            ax.set_xlabel("Perturbation" if i == n_rows - 1 else "")
            ax.set_ylabel(ylabel if j == 0 else "")

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_replicability_barplot(fr_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of replicability FR per modality/timepoint, hued by Cell."""
    _labelled_barplot(
        fr_df,
        output_path,
        label=fr_df["Modality"] + "_" + fr_df["time"],
        figsize=(12, 6),
        palette="Set2",
        xlabel="Modality / Timepoint",
        title="Replicability: Fraction Retrieved (q < 0.05)",
    )


def plot_matching_barplot(fr_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of target-matching FR per modality/timepoint, hued by Cell."""
    _labelled_barplot(
        fr_df,
        output_path,
        label=fr_df["Modality"] + "_" + fr_df["time"],
        figsize=(10, 6),
        palette="Set3",
        xlabel="Modality / Timepoint",
        title="Target Matching: Fraction Retrieved (q < 0.05)",
    )


def plot_cross_modality_barplot(fr_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of gene-compound matching FR per modality pair, hued by Cell."""
    if fr_df.empty:
        print("No cross-modality results to plot")
        return
    _labelled_barplot(
        fr_df,
        output_path,
        label=fr_df["Modality1"] + " vs " + fr_df["Modality2"],
        figsize=(12, 6),
        palette="Set1",
        xlabel="Modality Comparison",
        title="Cross-Modality Matching: Fraction Retrieved (q < 0.05)",
    )


def plot_replicability_map_boxplot(map_df: pd.DataFrame, output_path: Path) -> None:
    """Box plot of mAP values faceted by Cell (row) and time (col)."""
    if map_df.empty:
        print("No replicability mAP data to plot")
        return
    _faceted(
        map_df,
        output_path,
        plot=sns.boxplot,
        y="mean_average_precision",
        ylabel="mAP",
        palette="Set2",
        title="Replicability: mAP Distribution",
    )


def plot_matching_map_boxplot(map_df: pd.DataFrame, output_path: Path) -> None:
    """Box plot of matching mAP values faceted by Cell (row) and time (col)."""
    if map_df.empty:
        print("No matching mAP data to plot")
        return
    _faceted(
        map_df,
        output_path,
        plot=sns.boxplot,
        y="mean_average_precision",
        ylabel="mAP",
        palette="Set3",
        title="Target Matching: mAP Distribution",
    )


def plot_replicability_fr_faceted(fr_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of FR faceted by Cell (row) and time (col)."""
    if fr_df.empty:
        print("No replicability FR data to plot")
        return
    _faceted(
        fr_df,
        output_path,
        plot=sns.barplot,
        y="fr",
        ylabel="Fraction Retrieved",
        palette="Set2",
        title="Replicability: Fraction Retrieved",
    )


def plot_matching_fr_faceted(fr_df: pd.DataFrame, output_path: Path) -> None:
    """Bar plot of matching FR faceted by Cell (row) and time (col)."""
    if fr_df.empty:
        print("No matching FR data to plot")
        return
    _faceted(
        fr_df,
        output_path,
        plot=sns.barplot,
        y="fr",
        ylabel="Fraction Retrieved",
        palette="Set3",
        title="Target Matching: Fraction Retrieved",
    )
