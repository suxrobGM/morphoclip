"""Embedding-space diagnostic metrics for training monitoring.

Pure functions for computing alignment, uniformity, and other diagnostics
that help detect overfitting, embedding collapse, and training health issues.
"""

from collections.abc import Iterable

import torch
from torch import nn


@torch.no_grad()
def compute_alignment(
    image_emb: torch.Tensor,
    text_emb: torch.Tensor,
) -> float:
    """Mean cosine similarity of matched (diagonal) image-text pairs.

    Higher alignment means positive pairs are closer in embedding space.
    Saturating near 1.0 while eval loss rises signals overfitting.

    Args:
        image_emb: ``(B, D)`` L2-normalized image embeddings.
        text_emb: ``(B, D)`` L2-normalized text embeddings.

    Returns:
        Scalar alignment score in ``[-1, 1]``.
    """
    return float((image_emb * text_emb).sum(dim=1).mean().item())


@torch.no_grad()
def compute_uniformity(
    emb: torch.Tensor,
    *,
    t: float = 2.0,
) -> float:
    """Measure how uniformly embeddings spread on the unit hypersphere.

    Uses the metric from Wang & Isola (2020):
    ``log(mean(exp(-t * ||x_i - x_j||^2)))`` over all pairs.

    More negative values indicate better spread. Values approaching 0
    signal embedding collapse (all points mapping to the same region).

    Args:
        emb: ``(N, D)`` L2-normalized embeddings.
        t: Temperature parameter (default 2.0 per the paper).

    Returns:
        Scalar uniformity score (negative is better).
    """
    sq_pdist = torch.cdist(emb, emb, p=2).pow(2)
    # Exclude self-distances (diagonal)
    mask = ~torch.eye(emb.shape[0], dtype=torch.bool, device=emb.device)
    pairwise = sq_pdist[mask]
    return float(torch.exp(-t * pairwise).mean().log().item())


@torch.no_grad()
def compute_intra_batch_similarity(emb: torch.Tensor) -> float:
    """Mean off-diagonal cosine similarity within a batch.

    High values indicate embeddings are collapsing to similar
    representations regardless of input (mode collapse).

    Args:
        emb: ``(N, D)`` L2-normalized embeddings.

    Returns:
        Mean off-diagonal similarity.
    """
    sim = emb @ emb.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


@torch.no_grad()
def compute_logit_stats(logits: torch.Tensor) -> dict[str, float]:
    """Compute statistics of the similarity matrix.

    Useful for detecting temperature/scale issues. Collapsing std
    means the model cannot discriminate positive from negative pairs.

    Args:
        logits: ``(B, B)`` similarity matrix (after temperature scaling).

    Returns:
        Dict with ``mean``, ``std``, and ``max`` values.
    """
    return {
        "mean": float(logits.mean().item()),
        "std": float(logits.std().item()),
        "max": float(logits.max().item()),
    }


@torch.no_grad()
def compute_param_norm(module: nn.Module) -> float:
    """Total L2 norm of all parameters in a module.

    Args:
        module: Any ``nn.Module``.

    Returns:
        Scalar L2 norm.
    """
    total = 0.0
    for p in module.parameters():
        total += float(p.data.norm(2).item() ** 2)
    return total**0.5


def compute_grad_norm(params: Iterable[torch.Tensor | nn.Parameter]) -> float:
    """Total L2 norm of gradients across all parameters.

    Call this after ``clip_grad_norm_`` to get the post-clip norm.

    Args:
        params: Iterable of parameters (may include None grads).

    Returns:
        Scalar gradient L2 norm, or 0.0 if no gradients exist.
    """
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.data.norm(2).item() ** 2)
    return total**0.5
