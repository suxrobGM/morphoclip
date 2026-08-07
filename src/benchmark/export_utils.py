"""Shared helpers for exporting benchmark-layout profile CSVs.

Reused by both the CellCLIP local export pipeline
(``cellclip.benchmark.export``) and the MorphoCLIP profile exporter
(``morphoclip.benchmark.export``) so the on-disk layout the benchmark
harness expects (``benchmark.data.ProfileLoader``) stays in one place.
"""

from pathlib import Path

import numpy as np
import pandas as pd


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

    The benchmark consumes files named ``normalized_feature_select_negcon_batch``.
    Raw well embeddings typically carry a strong shared offset across wells, so we
    remove the negative-control reference mean before saving.

    Args:
        profiles: DataFrame with ``Metadata_*`` columns and feature columns.
        control_col: Column identifying negative-control wells.

    Returns:
        A copy of ``profiles`` with feature columns centered. Falls back to
        centering against all wells when no negative controls are present.
    """
    feature_cols = [col for col in profiles.columns if not col.startswith("Metadata")]
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
