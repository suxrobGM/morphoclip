"""Metrics for evaluating perturbation retrieval performance.

Backend loading lives in :mod:`benchmark.copairs_backend` and the old-copairs
("stable") pipeline in :mod:`benchmark.stable_map`; this module keeps the public
mAP API and dispatches between the stable and experimental backends.
"""

import numpy as np
import pandas as pd

from benchmark import copairs_backend, stable_map
from benchmark.copairs_backend import CopairsMode
from benchmark.data import get_features, get_metadata

__all__ = [
    "CopairsMode",
    "run_map_pipeline",
    "compute_map",
    "compute_fraction_retrieved",
    "evaluate_replicability",
    "evaluate_matching",
    "evaluate_cross_modality_matching",
]


def run_map_pipeline(
    meta: pd.DataFrame,
    features: np.ndarray,
    pos_sameby: list[str],
    pos_diffby: list[str],
    neg_sameby: list[str],
    neg_diffby: list[str],
    null_size: int = 100000,
    batch_size: int = 20000,
    use_abs: bool = False,
    multilabel_col: str | None = None,
    copairs_mode: CopairsMode = "experimental",
) -> pd.DataFrame:
    """Run the full mAP computation pipeline.

    Args:
        meta: Metadata DataFrame (will be reset to ensure consistent indexing).
        features: Feature matrix aligned with metadata rows.
        pos_sameby: Columns that must match for positive pairs.
        pos_diffby: Columns that must differ for positive pairs.
        neg_sameby: Columns that must match for negative pairs.
        neg_diffby: Columns that must differ for negative pairs.
        null_size: Number of permutations for null distribution.
        batch_size: Batch size for similarity computation.
        use_abs: Use absolute similarity (for anti-correlation matching).
        multilabel_col: Column with multi-label values.
        copairs_mode: "experimental" for new copairs API or "stable" for old API.
            Backward-compatible aliases: "modern" -> "experimental",
            "legacy" -> "stable".

    Returns:
        DataFrame with per-sample average precision statistics.
    """
    distance = "abs_cosine" if use_abs else "cosine"
    meta = meta.reset_index(drop=True).copy()

    mode = copairs_backend._normalize_copairs_mode(copairs_mode)
    if mode == "stable":
        return stable_map._run_legacy_pipeline(
            meta,
            features,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            null_size,
            batch_size,
            use_abs,
            multilabel_col,
        )

    modern = copairs_backend._get_modern_modules()

    if multilabel_col and multilabel_col in pos_sameby:
        return modern["average_precision_multilabel"](
            meta=meta,
            feats=features,  # type: ignore
            pos_sameby=pos_sameby,
            pos_diffby=pos_diffby,
            neg_sameby=neg_sameby,
            neg_diffby=neg_diffby,
            multilabel_col=multilabel_col,
            batch_size=batch_size,
            distance=distance,
            progress_bar=False,
        )

    return modern["average_precision_single"](
        meta=meta,
        feats=features,  # type: ignore
        pos_sameby=pos_sameby,
        pos_diffby=pos_diffby,
        neg_sameby=neg_sameby,
        neg_diffby=neg_diffby,
        batch_size=batch_size,
        distance=distance,
        progress_bar=False,
    )


def compute_map(
    result: pd.DataFrame,
    group_cols: list[str],
    threshold: float = 0.05,
    null_size: int = 100000,
    copairs_mode: CopairsMode = "experimental",
) -> pd.DataFrame:
    """Aggregate per-sample AP to mean Average Precision per group.

    Args:
        result: DataFrame with per-sample average precision values.
        group_cols: Columns to group by for aggregation.
        threshold: Q-value threshold for significance.
        copairs_mode: "experimental" for new copairs API or "stable" for old API.
            Backward-compatible aliases: "modern" -> "experimental",
            "legacy" -> "stable".

    Returns:
        DataFrame with mAP per group.
    """
    mode = copairs_backend._normalize_copairs_mode(copairs_mode)
    if mode == "stable":
        legacy = copairs_backend._get_legacy_modules()
        map_df = legacy["aggregate"](result, group_cols, threshold=threshold).rename(
            columns={"average_precision": "mean_average_precision"}
        )
        if "above_q_threshold" not in map_df.columns:
            if "above_corrected_p_threshold" in map_df.columns:
                map_df = map_df.rename(columns={"above_corrected_p_threshold": "above_q_threshold"})
            elif "below_corrected_p" in map_df.columns:
                map_df = map_df.rename(columns={"below_corrected_p": "above_q_threshold"})
        return map_df

    modern = copairs_backend._get_modern_modules()
    map_df = modern["mean_average_precision"](
        ap_scores=result,
        sameby=group_cols,
        null_size=null_size,
        threshold=threshold,
        seed=0,
        progress_bar=False,
    )
    return map_df.rename(columns={"below_corrected_p": "above_q_threshold"})


def compute_fraction_retrieved(map_df: pd.DataFrame) -> float:
    """Compute fraction of groups above significance threshold.

    Args:
        map_df: DataFrame with 'above_q_threshold' column from compute_map.

    Returns:
        Fraction of groups that are significant.
    """
    return len(map_df.query("above_q_threshold==True")) / len(map_df)


def evaluate_replicability(
    profiles: pd.DataFrame,
    sample_col: str = "Metadata_broad_sample",
    plate_col: str = "Metadata_Plate",
    negcon_col: str = "Metadata_negcon",
    null_size: int = 100000,
    batch_size: int = 20000,
    copairs_mode: CopairsMode = "experimental",
) -> pd.DataFrame:
    """Evaluate replicability mAP for profiles.

    Measures how well replicates of the same perturbation cluster together.

    Args:
        profiles: DataFrame with metadata and features.
        sample_col: Column identifying perturbation samples.
        plate_col: Column identifying plates.
        negcon_col: Binary column indicating negative controls.
        null_size: Permutations for null distribution.
        batch_size: Batch size for similarity computation.

    Returns:
        DataFrame with per-sample average precision values.
    """
    pos_sameby = [sample_col]
    pos_diffby = []
    neg_sameby = [plate_col]
    neg_diffby = [negcon_col]

    meta = get_metadata(profiles)
    features = get_features(profiles).values

    result = run_map_pipeline(
        meta,
        features,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        null_size=null_size,
        batch_size=batch_size,
        copairs_mode=copairs_mode,
    )

    if negcon_col not in result.columns:
        if negcon_col in meta.columns and len(meta) == len(result):
            result = result.copy()
            result[negcon_col] = meta[negcon_col].to_numpy()
        else:
            result = result.copy()
            result[negcon_col] = 0

    return result.query(f"{negcon_col}==0").reset_index(drop=True)


def evaluate_matching(
    profiles: pd.DataFrame,
    target_col: str = "Metadata_matching_target",
    use_abs: bool = True,
    multilabel: bool = True,
    null_size: int = 100000,
    batch_size: int = 20000,
    copairs_mode: CopairsMode = "experimental",
) -> pd.DataFrame:
    """Evaluate target matching mAP for consensus profiles.

    Measures how well perturbations with same target cluster together.

    Args:
        profiles: DataFrame with consensus profiles.
        target_col: Column with target annotations (may be list-valued).
        use_abs: Use absolute correlation for matching.
        multilabel: Whether target column contains multi-label values.
        null_size: Permutations for null distribution.
        batch_size: Batch size for similarity computation.

    Returns:
        DataFrame with per-sample average precision values.
    """
    pos_sameby = [target_col]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = [target_col]

    meta = get_metadata(profiles)
    features = get_features(profiles).values

    return run_map_pipeline(
        meta,
        features,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        null_size=null_size,
        batch_size=batch_size,
        use_abs=use_abs,
        multilabel_col=target_col if multilabel else None,
        copairs_mode=copairs_mode,
    )


def evaluate_cross_modality_matching(
    profiles: pd.DataFrame,
    target_col: str = "Metadata_matching_target",
    modality_col: str = "Metadata_modality",
    null_size: int = 100000,
    batch_size: int = 20000,
    copairs_mode: CopairsMode = "experimental",
) -> pd.DataFrame:
    """Evaluate cross-modality matching (e.g., compound-gene).

    Measures how well chemical and genetic perturbations with same target match.

    Args:
        profiles: DataFrame with profiles from multiple modalities.
        target_col: Column with target annotations.
        modality_col: Column indicating perturbation modality.
        null_size: Permutations for null distribution.
        batch_size: Batch size for similarity computation.

    Returns:
        DataFrame with per-sample average precision values.
    """
    pos_sameby = [target_col]
    pos_diffby = [modality_col]
    neg_sameby = []
    neg_diffby = [target_col, modality_col]

    meta = get_metadata(profiles)
    features = get_features(profiles).values

    return run_map_pipeline(
        meta,
        features,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        null_size=null_size,
        batch_size=batch_size,
        use_abs=True,
        multilabel_col=target_col,
        copairs_mode=copairs_mode,
    )
