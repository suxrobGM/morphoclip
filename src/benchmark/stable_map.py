"""Old-copairs ("stable") mAP pipeline for benchmark metrics.

Reaches :func:`benchmark.copairs_backend._get_legacy_modules` through the module
reference so tests can monkeypatch the backend loader.
"""

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd

from benchmark import copairs_backend
from benchmark.copairs_backend import _is_multiprocessing_permission_error


def _run_with_serial_fallback[T](primary: Callable[[], T], fallback: Callable[[], T]) -> T:
    """Run *primary*, falling back to *fallback* on multiprocessing PermissionErrors."""
    try:
        return primary()
    except Exception as exc:
        if not _is_multiprocessing_permission_error(exc):
            raise
        return fallback()


def _create_legacy_matcher(
    meta: pd.DataFrame,
    pos_sameby: list[str],
    pos_diffby: list[str],
    neg_sameby: list[str],
    neg_diffby: list[str],
    multilabel_col: str | None,
):
    legacy = copairs_backend._get_legacy_modules()
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
    legacy = copairs_backend._get_legacy_modules()
    dist_df = pairs[["ix1", "ix2"]].drop_duplicates().copy()
    if dist_df.empty:
        result = pairs.copy()
        result["dist"] = pd.Series(dtype=float)
        return result

    def _primary() -> np.ndarray:
        return legacy["cosine_indexed"](features, dist_df.values, batch_size)

    def _fallback() -> np.ndarray:
        pair_ix = dist_df[["ix1", "ix2"]].to_numpy(dtype=int, copy=False)
        corrs = []
        for start in range(0, len(pair_ix), batch_size):
            batch_ix = pair_ix[start : start + batch_size]
            x_sample = features[batch_ix[:, 0]]
            y_sample = features[batch_ix[:, 1]]
            corrs.append(legacy["backend"].pairwise_cosine(x_sample, y_sample))
        return np.concatenate(corrs) if corrs else np.array([], dtype=float)

    dist_df["dist"] = _run_with_serial_fallback(_primary, _fallback)
    if use_abs:
        dist_df["dist"] = np.abs(dist_df["dist"])
    return pairs.merge(dist_df, on=["ix1", "ix2"])


def _compute_legacy_null_dists(rel_k_list: pd.Series, null_size: int) -> np.ndarray:
    """Compute null distributions and fall back to serial sampling if Pool is unavailable."""
    legacy = copairs_backend._get_legacy_modules()

    def _primary() -> np.ndarray:
        return legacy["backend"].compute_null_dists(rel_k_list, null_size)

    def _fallback() -> np.ndarray:
        num_pos_list = rel_k_list.apply(np.sum)
        num_neg_list = rel_k_list.apply(np.size) - num_pos_list
        null_confs = [
            (null_size, int(num_pos), int(num_neg))
            for num_pos, num_neg in zip(num_pos_list, num_neg_list, strict=True)
        ]
        if not null_confs:
            return np.empty((0, null_size), dtype=float)
        return np.stack([legacy["backend"].random_ap(*key) for key in null_confs])

    return _run_with_serial_fallback(_primary, _fallback)


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
    legacy = copairs_backend._get_legacy_modules()
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
