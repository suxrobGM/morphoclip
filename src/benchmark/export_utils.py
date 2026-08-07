"""Shared helpers for exporting benchmark-layout profile CSVs.

Used by ``cellclip.benchmark.export`` and ``morphoclip.benchmark.export`` so the
on-disk layout ``benchmark.data.ProfileLoader`` reads stays in one place.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.data import get_feature_columns


def feature_columns(width: int) -> list[str]:
    """Build exported feature column names.

    Args:
        width: Number of feature dimensions.

    Returns:
        Column names ``feature_0000``, ``feature_0001``, ... up to ``width``.
    """
    return [f"feature_{i:04d}" for i in range(width)]


def negcon_center_profiles(
    profiles: pd.DataFrame,
    *,
    control_col: str = "Metadata_control_type",
) -> pd.DataFrame:
    """Center exported features against plate-level negative controls.

    Raw well embeddings carry a strong shared offset, so the negative-control
    mean is removed before saving.

    Args:
        profiles: DataFrame with ``Metadata_*`` columns and feature columns.
        control_col: Column identifying negative-control wells.

    Returns:
        A copy of ``profiles`` with feature columns centered. Falls back to
        centering against all wells when no negative controls are present.
    """
    feature_cols = get_feature_columns(profiles)
    if not feature_cols:
        return profiles

    negcon_mask = profiles[control_col].eq("negcon") if control_col in profiles.columns else None
    if negcon_mask is not None and bool(negcon_mask.any()):
        reference = profiles.loc[negcon_mask, feature_cols]
    else:
        reference = profiles[feature_cols]

    centered = profiles.copy()
    reference_mean = reference.to_numpy(dtype=np.float32).mean(axis=0)
    centered[feature_cols] = profiles[feature_cols].to_numpy(dtype=np.float32) - reference_mean
    return centered


def output_profile_path(output_profiles_root: Path, batch: str, plate: str) -> Path:
    """Return the benchmark-compatible exported profile path for a plate."""
    return (
        output_profiles_root
        / batch
        / plate
        / f"{plate}_normalized_feature_select_negcon_batch.csv.gz"
    )
