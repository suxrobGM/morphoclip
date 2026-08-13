"""The on-disk profile contract: where a plate's CSV lives and which columns are features.

Exporters and the benchmark harness share this layout, so it is defined once here.
Four separate predicates used to decide the metadata/feature split, and they disagreed.
Now :func:`metadata_columns` and :func:`feature_columns` are exact complements, so
every column lands in exactly one set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROFILE_SUFFIX = "normalized_feature_select_negcon_batch.csv.gz"

# `Meta_` is the alias CellCLIP's source profiles use for a handful of columns.
METADATA_PREFIXES = ("Metadata_", "Meta_")

CONTROL_COLUMN = "Metadata_control_type"
NEGCON_LABEL = "negcon"


def profile_path(profiles_root: Path, batch: str, plate: str) -> Path:
    """Path of a plate's profile CSV under *profiles_root*."""
    return profiles_root / batch / plate / f"{plate}_{PROFILE_SUFFIX}"


def metadata_columns(df: pd.DataFrame) -> list[str]:
    """Columns describing a well rather than measuring it."""
    return [c for c in df.columns if c.startswith(METADATA_PREFIXES)]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric columns, the exact complement of :func:`metadata_columns`."""
    return [c for c in df.columns if not c.startswith(METADATA_PREFIXES)]


def metadata_frame(df: pd.DataFrame) -> pd.DataFrame:
    """*df* narrowed to its metadata columns."""
    return df[metadata_columns(df)]


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """*df* narrowed to its feature columns."""
    return df[feature_columns(df)]


def numbered_feature_columns(width: int) -> list[str]:
    """Column names for a freshly exported embedding: ``feature_0000`` upward."""
    return [f"feature_{i:04d}" for i in range(width)]


def negcon_center_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Subtract the plate's negative-control mean from every feature column.

    Raw well embeddings carry a strong shared offset, so it is removed before
    saving. Plates with no negative controls fall back to centering against all
    wells rather than failing.

    Args:
        profiles: One plate's metadata and feature columns.

    Returns:
        A copy with feature columns centered, or *profiles* unchanged if it has
        no feature columns.
    """
    features = feature_columns(profiles)
    if not features:
        return profiles

    if CONTROL_COLUMN in profiles.columns:
        negcon = profiles[CONTROL_COLUMN].eq(NEGCON_LABEL)
        reference = profiles.loc[negcon, features] if negcon.any() else profiles[features]
    else:
        reference = profiles[features]

    centered = profiles.copy()
    reference_mean = reference.to_numpy(dtype=np.float32).mean(axis=0)
    centered[features] = profiles[features].to_numpy(dtype=np.float32) - reference_mean
    return centered


def write_profile(profiles: pd.DataFrame, path: Path) -> Path:
    """Write a plate profile to *path*, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(path, index=False, compression="gzip")
    return path
