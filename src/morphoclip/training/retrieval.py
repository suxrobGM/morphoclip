"""Retrieval metrics for MorphoCLIP evaluation.

**Why this module exists.** Replicate wells of the same perturbation share an
*identical* text embedding (the prompt is built from the perturbation, not the
well).  The validation split has ~2,220 wells drawn from only ~98 unique
perturbations, so scoring each image query against all N well-texts means
ranking a candidate pool that contains ~24 exact copies of every vector.
Identical scores clump those copies at consecutive ranks, which caps R@5/R@10
near R@1 and makes any random baseline meaningless.

**Fixed semantics.**

* *Well-level image->text*: each of the N wells ranks the **P unique** texts.
  Exactly one is correct (its own perturbation).  Pool size P, single positive.
* *Well-level text->image*: one query per unique perturbation (P queries) ranks
  all N well embeddings; the score is the rank of the *first* correct replicate
  (any-of-m retrieval with m = replicate count).
* *Perturbation-level*: per-perturbation mean image profiles (re-normalized)
  vs the P unique texts, a clean (P, P) problem with diagonal positives.

Analytic random baselines are emitted alongside every direction so a metric can
be read against chance without shuffling.
"""

import math

import torch
import torch.nn.functional as F

_KS: tuple[int, ...] = (1, 5, 10)


def dedupe_texts(
    text_features: torch.Tensor,
    broad_samples: list[str],
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Collapse per-well text embeddings to one row per unique perturbation.

    The first occurrence of each ``broad_sample`` wins; replicate rows are
    dropped (they are bit-identical copies in practice).

    Args:
        text_features: ``(N, D)`` per-well text embeddings.
        broad_samples: Perturbation ID per well (length N).

    Returns:
        Tuple of ``(unique_texts (P, D), unique_ids, well_to_pert (N,))`` where
        ``unique_ids`` is in first-occurrence order and ``well_to_pert`` is a
        ``LongTensor`` mapping each well to its row in ``unique_texts``.
    """
    id_to_idx: dict[str, int] = {}
    first_rows: list[int] = []
    assignments: list[int] = []
    for row, sample in enumerate(broad_samples):
        idx = id_to_idx.get(sample)
        if idx is None:
            idx = len(first_rows)
            id_to_idx[sample] = idx
            first_rows.append(row)
        assignments.append(idx)

    index = torch.tensor(first_rows, dtype=torch.long, device=text_features.device)
    well_to_pert = torch.tensor(assignments, dtype=torch.long, device=text_features.device)
    return text_features.index_select(0, index), list(id_to_idx), well_to_pert


def aggregate_images(
    image_features: torch.Tensor,
    broad_samples: list[str],
) -> torch.Tensor:
    """Average image embeddings per perturbation and re-normalize.

    Args:
        image_features: ``(N, D)`` per-well image embeddings.
        broad_samples: Perturbation ID per well (length N).

    Returns:
        ``(P, D)`` L2-normalized mean profiles, ordered to match the
        first-occurrence order produced by :func:`dedupe_texts`.
    """
    id_to_idx: dict[str, int] = {}
    assignments: list[int] = []
    for sample in broad_samples:
        idx = id_to_idx.get(sample)
        if idx is None:
            idx = len(id_to_idx)
            id_to_idx[sample] = idx
        assignments.append(idx)

    n_pert = len(id_to_idx)
    device = image_features.device
    dtype = image_features.dtype
    index = torch.tensor(assignments, dtype=torch.long, device=device)

    sums = torch.zeros(n_pert, image_features.shape[1], dtype=dtype, device=device)
    sums.index_add_(0, index, image_features)
    counts = torch.zeros(n_pert, dtype=dtype, device=device)
    counts.index_add_(0, index, torch.ones(len(assignments), dtype=dtype, device=device))
    return F.normalize(sums / counts.unsqueeze(1), dim=-1)


def _first_positive_rank(scores: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    """Rank (1-based) of the highest-scoring positive for each query row."""
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    ordered = positives.gather(1, order)
    return torch.argmax(ordered.to(torch.int64), dim=1) + 1


def _rank_metrics(prefix: str, scores: torch.Tensor, positives: torch.Tensor) -> dict[str, float]:
    """R@k plus mean/median rank for one retrieval direction."""
    ranks = _first_positive_rank(scores, positives).float()
    results: dict[str, float] = {
        f"{prefix}_mean_rank": float(ranks.mean().item()),
        f"{prefix}_median_rank": float(ranks.median().item()),
    }
    for k in _KS:
        results[f"{prefix}_R@{k}"] = float((ranks <= k).float().mean().item())
    return results


def _single_positive_random(prefix: str, n_candidates: int) -> dict[str, float]:
    """Chance performance for one positive in a pool of ``n_candidates``.

    ``R@k = min(k, P) / P``; the mean and median rank both use the continuous
    approximation ``(P + 1) / 2`` (the exact discrete median is
    ``ceil(P / 2)``, which differs by at most half a rank).
    """
    pool = float(n_candidates)
    results: dict[str, float] = {
        f"{prefix}_random_mean_rank": (pool + 1.0) / 2.0,
        f"{prefix}_random_median_rank": (pool + 1.0) / 2.0,
    }
    for k in _KS:
        results[f"{prefix}_random_R@{k}"] = min(float(k), pool) / pool
    return results


def _any_of_m_random(prefix: str, n_candidates: int, replicates: list[int]) -> dict[str, float]:
    """Chance performance for any-of-m retrieval, averaged over queries.

    For a query with ``m`` positives among ``N`` candidates the probability that
    a random top-k contains at least one positive is
    ``1 - C(N - m, k) / C(N, k)`` (1.0 once ``k > N - m``), and the expected
    rank of the first positive is ``(N + 1) / (m + 1)``.
    """
    n_queries = max(1, len(replicates))
    recalls = dict.fromkeys(_KS, 0.0)
    mean_rank = 0.0
    for m in replicates:
        negatives = n_candidates - m
        for k in _KS:
            if k > negatives:
                recalls[k] += 1.0
            else:
                recalls[k] += 1.0 - math.comb(negatives, k) / math.comb(n_candidates, k)
        mean_rank += (n_candidates + 1.0) / (m + 1.0)

    results: dict[str, float] = {f"{prefix}_random_mean_rank": mean_rank / n_queries}
    for k in _KS:
        results[f"{prefix}_random_R@{k}"] = recalls[k] / n_queries
    return results


def _diagonal_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> dict[str, float]:
    """Legacy diagonal-only retrieval over an ``(N, N)`` score matrix."""
    logits = image_features @ text_features.t()
    positives = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    return {
        **_rank_metrics("image_to_text", logits, positives),
        **_rank_metrics("text_to_image", logits.t(), positives.t()),
    }


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    broad_samples: list[str] | None = None,
) -> dict[str, float]:
    """Compute well-level and perturbation-level retrieval metrics.

    Both inputs are L2-normalized defensively (CWA batch correction in the eval
    loop can leave embeddings off the unit sphere).

    When ``broad_samples`` is ``None`` the legacy diagonal-only behavior is
    used: N images vs N texts, positive = diagonal, no perturbation-level or
    random-baseline keys.  That path exists for synthetic callers and tests.

    Args:
        image_features: ``(N, D)`` image embeddings.
        text_features: ``(N, D)`` per-well text embeddings.
        broad_samples: Perturbation ID per well (length N).

    Returns:
        Dict of plain floats: R@1/5/10 and mean/median rank for well-level
        ``image_to_text`` / ``text_to_image``, the ``pert_``-prefixed
        perturbation-level equivalents, matching ``*_random_*`` baselines, plus
        ``n_wells`` and ``n_perturbations``.
    """
    image_features = F.normalize(image_features.float(), dim=-1)
    text_features = F.normalize(text_features.float(), dim=-1)

    if broad_samples is None:
        return _diagonal_metrics(image_features, text_features)

    unique_texts, unique_ids, well_to_pert = dedupe_texts(text_features, broad_samples)
    n_wells = image_features.shape[0]
    n_pert = len(unique_ids)

    well_positives = F.one_hot(well_to_pert, num_classes=n_pert).bool()  # (N, P)
    i2t_scores = image_features @ unique_texts.t()  # (N, P)
    t2i_scores = unique_texts @ image_features.t()  # (P, N)

    pert_images = aggregate_images(image_features, broad_samples)
    pert_scores = pert_images @ unique_texts.t()  # (P, P)
    pert_positives = torch.eye(n_pert, dtype=torch.bool, device=pert_scores.device)

    replicates = torch.bincount(well_to_pert, minlength=n_pert).tolist()

    return {
        **_rank_metrics("image_to_text", i2t_scores, well_positives),
        **_rank_metrics("text_to_image", t2i_scores, well_positives.t()),
        **_rank_metrics("pert_image_to_text", pert_scores, pert_positives),
        **_rank_metrics("pert_text_to_image", pert_scores.t(), pert_positives),
        **_single_positive_random("image_to_text", n_pert),
        **_any_of_m_random("text_to_image", n_wells, replicates),
        **_single_positive_random("pert_image_to_text", n_pert),
        **_single_positive_random("pert_text_to_image", n_pert),
        "n_wells": float(n_wells),
        "n_perturbations": float(n_pert),
    }
