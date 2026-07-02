"""Data loading and processing utilities for benchmark evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Return list of metadata columns (prefixed with 'Metadata_')."""
    return [c for c in df.columns if c.startswith("Metadata_")]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature columns (not prefixed with 'Metadata')."""
    return [c for c in df.columns if not c.startswith("Metadata")]


def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe containing only metadata columns."""
    return df[get_metadata_columns(df)]


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe containing only feature columns."""
    return df[get_feature_columns(df)]


def normalize_subset_label(subset: str) -> str:
    """Normalize subset aliases to manifest labels."""
    normalized = str(subset).strip().lower()
    if normalized == "val":
        return "validate"
    if normalized in {"train", "validate", "test"}:
        return normalized
    raise ValueError(f"Unknown split subset: {subset!r}")


def load_split_manifest(path: str | Path, subset: str) -> pd.DataFrame:
    """Load a split manifest and return rows for one subset."""
    path = Path(path)
    manifest = pd.read_csv(path)
    required = {"subset", "Metadata_Plate", "Metadata_Well"}
    missing = required - set(manifest.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Split manifest is missing required columns: {missing_display}")

    normalized_subset = normalize_subset_label(subset)
    manifest["subset"] = manifest["subset"].map(normalize_subset_label)
    filtered = manifest[manifest["subset"] == normalized_subset].copy()
    if filtered.empty:
        raise ValueError(
            f"Split manifest {path} does not contain any rows for subset {normalized_subset!r}"
        )

    filtered["Metadata_Plate"] = filtered["Metadata_Plate"].astype(str)
    filtered["Metadata_Well"] = filtered["Metadata_Well"].astype(str)
    return filtered.reset_index(drop=True)


def filter_profiles_to_split_subset(
    df: pd.DataFrame,
    split_manifest: pd.DataFrame | None,
    *,
    keep_negcon_on_selected_plates: bool = False,
    control_col: str = "Metadata_control_type",
    negcon_value: str = "negcon",
) -> pd.DataFrame:
    """Keep only profile rows whose ``Metadata_Plate`` + ``Metadata_Well`` are in the manifest."""
    if split_manifest is None or df.empty:
        return df

    required = {"Metadata_Plate", "Metadata_Well"}
    missing = required - set(df.columns)
    if missing:
        missing_display = ", ".join(sorted(missing))
        raise ValueError(f"Profile dataframe is missing required split columns: {missing_display}")

    subset_pairs = split_manifest.loc[:, ["Metadata_Plate", "Metadata_Well"]].drop_duplicates()
    selected_plates = set(subset_pairs["Metadata_Plate"].astype(str))
    merged = df.merge(
        subset_pairs.assign(_keep=1),
        on=["Metadata_Plate", "Metadata_Well"],
        how="left",
    )
    keep_mask = merged["_keep"] == 1
    if keep_negcon_on_selected_plates and control_col in merged.columns:
        keep_mask |= merged["Metadata_Plate"].astype(str).isin(selected_plates) & (
            merged[control_col] == negcon_value
        )

    return merged[keep_mask].drop(columns="_keep").reset_index(drop=True)


def filter_experiment_metadata_to_split_subset(
    df: pd.DataFrame,
    split_manifest: pd.DataFrame | None,
) -> pd.DataFrame:
    """Reduce experiment metadata to plates present in the selected split subset."""
    if split_manifest is None or df.empty:
        return df

    selected_plates = set(split_manifest["Metadata_Plate"].astype(str))
    filtered = df[df["Assay_Plate_Barcode"].astype(str).isin(selected_plates)].copy()
    return filtered.reset_index(drop=True)


class ProfileLoader:
    """Handles loading and preprocessing of Cell Painting profiles."""

    def __init__(self, profiles_dir: str | Path):
        """Initialize loader with base profiles directory.

        Args:
            profiles_dir: Path to directory containing batch/plate profiles.
        """
        self.profiles_dir = Path(profiles_dir)

    def load_plate(
        self,
        batch: str,
        plate: str,
        file_pattern: str = "normalized_feature_select_negcon_batch.csv.gz",
    ) -> pd.DataFrame:
        """Load profiles for a single plate.

        Args:
            batch: Batch identifier (e.g., '2020_11_04_CPJUMP1').
            plate: Plate barcode.
            file_pattern: Glob pattern for profile files.

        Returns:
            DataFrame with all profiles from matching files.
        """
        plate_dir = self.profiles_dir / batch / plate
        files = list(plate_dir.glob(f"*_{file_pattern}"))

        if not files:
            raise FileNotFoundError(f"No files matching pattern in {plate_dir}")

        dfs = [pd.read_csv(f, low_memory=False) for f in files]
        return pd.concat(dfs, ignore_index=True)

    def load_plates(
        self,
        batch: str,
        plates: Sequence[str],
        file_pattern: str = "normalized_feature_select_negcon_batch.csv.gz",
        modality: str | None = None,
    ) -> pd.DataFrame:
        """Load and concatenate profiles from multiple plates.

        Args:
            batch: Batch identifier.
            plates: List of plate barcodes.
            file_pattern: Glob pattern for profile files.
            modality: Optional modality label to assign.

        Returns:
            Concatenated DataFrame with all plate profiles.
        """
        dfs = []
        for plate in plates:
            df = self.load_plate(batch, plate, file_pattern)
            if modality:
                df["Metadata_modality"] = modality
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)


def remove_empty_wells(df: pd.DataFrame, sample_col: str = "Metadata_broad_sample") -> pd.DataFrame:
    """Remove wells with missing sample identifier."""
    return df.dropna(subset=[sample_col]).reset_index(drop=True)


def remove_negcon_wells(
    df: pd.DataFrame,
    control_col: str = "Metadata_control_type",
    sample_col: str = "Metadata_broad_sample",
) -> pd.DataFrame:
    """Remove negative control and empty wells."""
    return df.query(f'{control_col}!="negcon"').dropna(subset=[sample_col]).reset_index(drop=True)


def add_negcon_indicator(
    df: pd.DataFrame,
    control_col: str = "Metadata_control_type",
    indicator_col: str = "Metadata_negcon",
) -> pd.DataFrame:
    """Add binary indicator column for negative controls."""
    df = df.copy()
    df[indicator_col] = np.where(df[control_col] == "negcon", 1, 0)
    return df


def compute_consensus(
    df: pd.DataFrame,
    group_col: str,
    agg_func: str = "median",
) -> pd.DataFrame:
    """Compute consensus profiles by grouping and aggregating features.

    Args:
        df: DataFrame with metadata and feature columns.
        group_col: Column to group by for consensus.
        agg_func: Aggregation function ('median' or 'mean').

    Returns:
        DataFrame with one consensus profile per group.
    """
    metadata_df = get_metadata(df).drop_duplicates(subset=[group_col])
    assert metadata_df is not None, "metadata_df should not be None"
    feature_cols = [group_col] + get_feature_columns(df)

    if agg_func == "median":
        consensus_df = df[feature_cols].groupby(group_col).median().reset_index()
    else:
        consensus_df = df[feature_cols].groupby(group_col).mean().reset_index()

    return metadata_df.merge(consensus_df, on=group_col)


def filter_replicable(
    df: pd.DataFrame,
    replicable_ids: Sequence[str],
    id_col: str = "Metadata_broad_sample",
) -> pd.DataFrame:
    """Filter to keep only replicable perturbations."""
    return df.query(f"{id_col}==@replicable_ids").reset_index(drop=True)


def get_timepoint_label(modality: str, hours: int) -> str:
    """Convert timepoint in hours to short/long label.

    Args:
        modality: Perturbation type ('compound', 'orf', 'crispr').
        hours: Timepoint in hours.

    Returns:
        'short' or 'long' based on modality-specific thresholds.
    """
    thresholds = {"compound": 24, "orf": 48, "crispr": 96}
    threshold = thresholds.get(modality, 96)
    return "short" if hours <= threshold else "long"
