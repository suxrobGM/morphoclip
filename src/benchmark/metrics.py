"""Metrics for evaluating perturbation retrieval performance."""

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from benchmark.data import get_features, get_metadata

# Canonical modes:
# - stable: old copairs API (paper-compatible behavior)
# - experimental: new copairs API
# Backward-compatible aliases:
# - legacy -> stable
# - modern -> experimental
CopairsMode = Literal["stable", "experimental", "legacy", "modern"]

EXPERIMENTAL_COPAIRS_ERROR = (
    "Experimental copairs mode requires the new copairs API. Install a recent copairs release."
)

STABLE_COPAIRS_ERROR = (
    "Stable copairs mode requires the old copairs API. "
    "Install the reference version from the paper environment:\n"
    "pip install git+https://github.com/cytomining/copairs@"
    "880f22a551bd897896d148a0b07baa99d981c6a9"
)


@dataclass
class MAPResult:
    """Container for mAP evaluation results."""

    map_df: pd.DataFrame
    fraction_retrieved: float
    description: str


def _is_multiprocessing_permission_error(exc: Exception) -> bool:
    """Detect sandbox/runtime failures from old copairs multiprocessing helpers."""
    if isinstance(exc, PermissionError):
        return True

    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, PermissionError):
        return True

    message = str(exc).lower()
    return "semlock" in message and "permission denied" in message


def _normalize_copairs_mode(copairs_mode: CopairsMode) -> Literal["stable", "experimental"]:
    """Normalize copairs mode and keep backward compatibility."""
    if copairs_mode in ("stable", "legacy"):
        return "stable"
    if copairs_mode in ("experimental", "modern"):
        return "experimental"
    raise ValueError(
        f"Unsupported copairs_mode='{copairs_mode}'. "
        "Use one of: stable, experimental (aliases: legacy, modern)."
    )


def _get_modern_modules():
    """Import modern copairs modules required for experimental mode."""
    try:
        from copairs.map import mean_average_precision  # type: ignore
        from copairs.map.average_precision import (
            average_precision as average_precision_single,  # type: ignore
        )
        from copairs.map.multilabel import (
            average_precision as average_precision_multilabel,  # type: ignore
        )
    except Exception as exc:  # pragma: no cover - depends on installed copairs version
        raise RuntimeError(EXPERIMENTAL_COPAIRS_ERROR) from exc

    return {
        "mean_average_precision": mean_average_precision,
        "average_precision_single": average_precision_single,
        "average_precision_multilabel": average_precision_multilabel,
    }


def _get_legacy_modules():
    """Import old copairs modules required for stable mode."""
    try:
        import copairs.compute_np as backend  # type: ignore
        from copairs.compute import cosine_indexed  # type: ignore
        from copairs.map import (
            aggregate,  # type: ignore
            build_rank_list_multi,  # type: ignore
            build_rank_lists,  # type: ignore
            results_to_dframe,  # type: ignore
        )
        from copairs.matching import Matcher, MatcherMultilabel, dict_to_dframe
    except Exception as exc:  # pragma: no cover - depends on installed copairs version
        raise RuntimeError(STABLE_COPAIRS_ERROR) from exc

    return {
        "backend": backend,
        "cosine_indexed": cosine_indexed,
        "aggregate": aggregate,
        "build_rank_list_multi": build_rank_list_multi,
        "build_rank_lists": build_rank_lists,
        "results_to_dframe": results_to_dframe,
        "Matcher": Matcher,
        "MatcherMultilabel": MatcherMultilabel,
        "dict_to_dframe": dict_to_dframe,
    }


def _create_legacy_matcher(
    meta: pd.DataFrame,
    pos_sameby: list[str],
    pos_diffby: list[str],
    neg_sameby: list[str],
    neg_diffby: list[str],
    multilabel_col: str | None,
):
    legacy = _get_legacy_modules()
    columns = list(set(pos_sameby + pos_diffby + neg_sameby + neg_diffby))
    sameby_cols = list(set(pos_sameby) | set(neg_sameby))
    multilabel_col = multilabel_col if multilabel_col in sameby_cols else None

    if multilabel_col is not None:
        return legacy["MatcherMultilabel"](meta, columns, seed=0, multilabel_col=multilabel_col)
    return legacy["Matcher"](meta, columns, seed=0)


def _compute_legacy_similarities(
    pairs: pd.DataFrame,
    features: np.ndarray,
    batch_size: int,
    use_abs: bool,
) -> pd.DataFrame:
    legacy = _get_legacy_modules()
    dist_df = pairs[["ix1", "ix2"]].drop_duplicates().copy()
    if dist_df.empty:
        result = pairs.copy()
        result["dist"] = pd.Series(dtype=float)
        return result

    try:
        dist_df["dist"] = legacy["cosine_indexed"](features, dist_df.values, batch_size)
    except Exception as exc:
        if not _is_multiprocessing_permission_error(exc):
            raise

        pair_ix = dist_df[["ix1", "ix2"]].to_numpy(dtype=int, copy=False)
        corrs = []
        for start in range(0, len(pair_ix), batch_size):
            batch_ix = pair_ix[start : start + batch_size]
            x_sample = features[batch_ix[:, 0]]
            y_sample = features[batch_ix[:, 1]]
            corrs.append(legacy["backend"].pairwise_cosine(x_sample, y_sample))
        dist_df["dist"] = np.concatenate(corrs) if corrs else np.array([], dtype=float)
    if use_abs:
        dist_df["dist"] = np.abs(dist_df["dist"])
    return pairs.merge(dist_df, on=["ix1", "ix2"])


def _compute_legacy_null_dists(rel_k_list: pd.Series, null_size: int) -> np.ndarray:
    """Compute null distributions and fall back to serial sampling if Pool is unavailable."""
    legacy = _get_legacy_modules()
    try:
        return legacy["backend"].compute_null_dists(rel_k_list, null_size)
    except Exception as exc:
        if not _is_multiprocessing_permission_error(exc):
            raise

        num_pos_list = rel_k_list.apply(np.sum)
        num_neg_list = rel_k_list.apply(np.size) - num_pos_list
        null_confs = [
            (null_size, int(num_pos), int(num_neg))
            for num_pos, num_neg in zip(num_pos_list, num_neg_list, strict=True)
        ]
        if not null_confs:
            return np.empty((0, null_size), dtype=float)
        return np.stack([legacy["backend"].random_ap(*key) for key in null_confs])


def _run_legacy_pipeline(
    meta: pd.DataFrame,
    features: np.ndarray,
    pos_sameby: list[str],
    pos_diffby: list[str],
    neg_sameby: list[str],
    neg_diffby: list[str],
    null_size: int,
    batch_size: int,
    use_abs: bool,
    multilabel_col: str | None,
) -> pd.DataFrame:
    legacy = _get_legacy_modules()
    matcher = _create_legacy_matcher(
        meta, pos_sameby, pos_diffby, neg_sameby, neg_diffby, multilabel_col
    )

    dict_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
    pos_pairs = legacy["dict_to_dframe"](dict_pairs, pos_sameby)
    dict_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
    neg_pairs = set(itertools.chain.from_iterable(dict_pairs.values()))
    neg_pairs = pd.DataFrame(neg_pairs, columns=["ix1", "ix2"])

    pos_pairs = _compute_legacy_similarities(pos_pairs, features, batch_size, use_abs)
    neg_pairs = _compute_legacy_similarities(neg_pairs, features, batch_size, use_abs)
    if pos_pairs.empty or neg_pairs.empty:
        return pd.DataFrame()

    if multilabel_col and multilabel_col in pos_sameby:
        rel_k_list = legacy["build_rank_list_multi"](pos_pairs, neg_pairs, multilabel_col)
    else:
        rel_k_list = legacy["build_rank_lists"](pos_pairs, neg_pairs)
    if rel_k_list.empty:
        return pd.DataFrame()

    ap_scores = rel_k_list.apply(legacy["backend"].compute_ap)
    if ap_scores.empty:
        return pd.DataFrame()
    ap_scores = np.concatenate(ap_scores.values)
    null_dists = _compute_legacy_null_dists(rel_k_list, null_size)
    p_values = legacy["backend"].compute_p_values(null_dists, ap_scores, null_size)

    return legacy["results_to_dframe"](meta, rel_k_list.index, p_values, ap_scores, multilabel_col)


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

    mode = _normalize_copairs_mode(copairs_mode)
    if mode == "stable":
        return _run_legacy_pipeline(
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

    modern = _get_modern_modules()

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
    mode = _normalize_copairs_mode(copairs_mode)
    if mode == "stable":
        legacy = _get_legacy_modules()
        map_df = legacy["aggregate"](result, group_cols, threshold=threshold).rename(
            columns={"average_precision": "mean_average_precision"}
        )
        if "above_q_threshold" not in map_df.columns:
            if "above_corrected_p_threshold" in map_df.columns:
                map_df = map_df.rename(columns={"above_corrected_p_threshold": "above_q_threshold"})
            elif "below_corrected_p" in map_df.columns:
                map_df = map_df.rename(columns={"below_corrected_p": "above_q_threshold"})
        return map_df

    modern = _get_modern_modules()
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
