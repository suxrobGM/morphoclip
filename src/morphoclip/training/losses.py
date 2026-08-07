"""Loss functions for MorphoCLIP training.

Provides InfoNCE (standard symmetric CLIP loss), CWCL (Continuously
Weighted Contrastive Loss with perturbation-identity soft labels), and an
optional replicate-alignment image-image term.

Note on naming: CellCLIP uses "CWCL" for *Channel-Wise* Contrastive Loss
(soft labels from per-channel image similarity).  MorphoCLIP's CWCL is
*Continuously Weighted* — soft labels come from perturbation identity
(``broad_sample``) and, optionally, shared target genes.
"""

import torch
import torch.nn.functional as F

from morphoclip.data.perturbation import parse_target_gene_key


def infonce_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP) loss.

    Diagonal entries are treated as positive pairs.

    Args:
        image_features: ``(B, D)`` L2-normalized image embeddings.
        text_features: ``(B, D)`` L2-normalized text embeddings.
        logit_scale: Scalar log-temperature parameter.

    Returns:
        Scalar loss (mean of image-to-text and text-to-image CE).
    """
    logits = logit_scale.exp() * image_features @ text_features.t()
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.t(), targets)
    return (loss_i2t + loss_t2i) / 2


def build_affinity_matrix(
    broad_samples: list[str],
    *,
    target_keys: list[str] | None = None,
    target_weight: float = 0.0,
    device: torch.device,
) -> torch.Tensor:
    """Build the un-normalized pairwise affinity matrix.

    Affinity is ``1.0`` for wells sharing a ``broad_sample``, *target_weight*
    for wells whose canonical gene sets intersect, and ``0`` otherwise.  The
    diagonal is always ``1.0``.  An empty gene key never matches anything.

    Args:
        broad_samples: Perturbation ID per sample (length B).
        target_keys: Optional canonical gene keys per sample, from
            :func:`morphoclip.data.perturbation.target_gene_key`.
        target_weight: Affinity given to gene-overlapping pairs.  ``0.0``
            (default) disables gene-aware weighting entirely.
        device: Target device.

    Returns:
        Symmetric affinity matrix ``(B, B)``.
    """
    unique = {s: idx for idx, s in enumerate(dict.fromkeys(broad_samples))}
    ids = torch.tensor([unique[s] for s in broad_samples], device=device)
    affinity = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()

    if target_keys is not None and target_weight > 0.0:
        if len(target_keys) != len(broad_samples):
            raise ValueError("target_keys must have the same length as broad_samples")
        gene_to_rows: dict[str, list[int]] = {}
        for row, key in enumerate(target_keys):
            for gene in parse_target_gene_key(key):
                gene_to_rows.setdefault(gene, []).append(row)

        overlap = torch.zeros_like(affinity)
        for rows in gene_to_rows.values():
            idx = torch.tensor(rows, device=device)
            overlap[idx.unsqueeze(1), idx.unsqueeze(0)] = target_weight
        affinity = torch.maximum(affinity, overlap)

    affinity.fill_diagonal_(1.0)
    return affinity


def _row_normalize(affinity: torch.Tensor) -> torch.Tensor:
    """Normalize each row of an affinity matrix to sum to 1."""
    row_sums = affinity.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return affinity / row_sums


def build_soft_labels(
    broad_samples: list[str],
    *,
    target_keys: list[str] | None = None,
    target_weight: float = 0.0,
    device: torch.device,
) -> torch.Tensor:
    """Build the row-normalized soft label matrix from perturbation identity.

    Samples sharing the same ``broad_sample`` get equal positive weight.
    When *target_weight* is positive, samples whose target gene sets
    intersect additionally get a partial weight (a compound and a CRISPR
    knockout hitting the same gene are soft positives).

    With ``target_weight=0.0`` this reduces exactly to the original binary
    same-``broad_sample`` behavior.

    Args:
        broad_samples: Perturbation ID per sample (length B).
        target_keys: Optional canonical gene keys per sample.
        target_weight: Affinity for gene-overlapping pairs (0 disables).
        device: Target device.

    Returns:
        Soft label matrix ``(B, B)``, rows sum to 1.
    """
    affinity = build_affinity_matrix(
        broad_samples,
        target_keys=target_keys,
        target_weight=target_weight,
        device=device,
    )
    return _row_normalize(affinity)


def cwcl_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    broad_samples: list[str],
    target_keys: list[str] | None = None,
    target_weight: float = 0.0,
) -> torch.Tensor:
    """Continuously Weighted Contrastive Loss with perturbation soft labels.

    Instead of hard diagonal targets, uses a soft label matrix where all
    wells sharing the same perturbation (``broad_sample``) — and optionally
    wells sharing a target gene — are positive pairs.

    Each direction normalizes its own target rows: the i2t targets are the
    rows of the affinity matrix ``A``, the t2i targets are the rows of
    ``A.t()``.  With purely binary labels these coincide with normalizing
    ``A`` once and transposing (group members share a row sum), so this is
    behavior-preserving; with continuous weights, transposing a
    row-normalized matrix would leave rows that do not sum to 1.

    Args:
        image_features: ``(B, D)`` L2-normalized image embeddings.
        text_features: ``(B, D)`` L2-normalized text embeddings.
        logit_scale: Scalar log-temperature parameter.
        broad_samples: Perturbation ID per sample (length B).
        target_keys: Optional canonical gene keys per sample.
        target_weight: Affinity for gene-overlapping pairs (0 disables).

    Returns:
        Scalar loss.
    """
    logits = logit_scale.exp() * image_features @ text_features.t()
    affinity = build_affinity_matrix(
        broad_samples,
        target_keys=target_keys,
        target_weight=target_weight,
        device=logits.device,
    )
    labels_i2t = _row_normalize(affinity)
    labels_t2i = _row_normalize(affinity.t())

    # Soft cross-entropy: -sum(W * log_softmax(S)) per row, averaged
    log_probs_i2t = F.log_softmax(logits, dim=1)
    log_probs_t2i = F.log_softmax(logits.t(), dim=1)

    loss_i2t = -(labels_i2t * log_probs_i2t).sum(dim=1).mean()
    loss_t2i = -(labels_t2i * log_probs_t2i).sum(dim=1).mean()

    return (loss_i2t + loss_t2i) / 2


def replicate_image_loss(
    image_features: torch.Tensor,
    scale: torch.Tensor | float,
    *,
    broad_samples: list[str],
) -> torch.Tensor:
    """Image-image contrastive loss pulling replicate wells together.

    Wells sharing a ``broad_sample`` are positives for each other; the
    self-similarity diagonal is excluded from both numerator and
    denominator.  Rows with no replicate in the batch contribute nothing
    and are dropped from the mean.

    The similarity matrix is symmetric, so a single direction suffices.

    Args:
        image_features: ``(B, D)`` L2-normalized image embeddings.
        scale: *Linear* logit scale — i.e. ``logit_scale.exp()`` for the
            shared learnable temperature, or ``1 / temperature`` for a
            fixed one.
        broad_samples: Perturbation ID per sample (length B).

    Returns:
        Scalar loss.  When no row has a replicate, returns a zero that is
        still attached to the autograd graph (keeps DDP gradients in sync).
    """
    device = image_features.device
    n = image_features.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=device)

    unique = {s: idx for idx, s in enumerate(dict.fromkeys(broad_samples))}
    ids = torch.tensor([unique[s] for s in broad_samples], device=device)
    positives = (ids.unsqueeze(0) == ids.unsqueeze(1)) & ~eye

    positive_counts = positives.sum(dim=1)
    valid = positive_counts > 0
    if not bool(valid.any()):
        return image_features.sum() * 0.0

    logits = scale * image_features @ image_features.t()
    logits = logits.masked_fill(eye, float("-inf"))
    log_probs = F.log_softmax(logits, dim=1).masked_fill(eye, 0.0)

    weights = positives.float() / positive_counts.clamp(min=1).unsqueeze(1)
    per_row = -(weights * log_probs).sum(dim=1)
    return per_row[valid].mean()


def compute_loss(
    loss_type: str,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    broad_samples: list[str] | None = None,
    target_keys: list[str] | None = None,
    target_weight: float = 0.0,
) -> torch.Tensor:
    """Dispatch to the appropriate loss function.

    Args:
        loss_type: ``"infonce"`` or ``"cwcl"``.
        image_features: ``(B, D)`` L2-normalized image embeddings.
        text_features: ``(B, D)`` L2-normalized text embeddings.
        logit_scale: Scalar log-temperature parameter.
        broad_samples: Required for CWCL — perturbation ID per sample.
        target_keys: Optional canonical gene keys per sample (CWCL only).
        target_weight: Affinity for gene-overlapping pairs (CWCL only).

    Returns:
        Scalar loss value.

    Raises:
        ValueError: If *loss_type* is unknown or *broad_samples* is
            missing for CWCL.
    """
    normalized = loss_type.strip().lower()
    if normalized == "infonce":
        return infonce_loss(image_features, text_features, logit_scale)
    if normalized == "cwcl":
        if broad_samples is None:
            raise ValueError("CWCL loss requires broad_samples")
        return cwcl_loss(
            image_features,
            text_features,
            logit_scale,
            broad_samples=broad_samples,
            target_keys=target_keys,
            target_weight=target_weight,
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
