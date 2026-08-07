"""Loss functions for MorphoCLIP training.

Provides InfoNCE (standard symmetric CLIP loss), CWCL (Continuously
Weighted Contrastive Loss with perturbation-identity soft labels), and an
optional replicate-alignment image-image term.

Naming: CellCLIP's "CWCL" is Channel-Wise Contrastive Loss (soft labels from
per-channel image similarity). MorphoCLIP's is Continuously Weighted, with soft
labels from perturbation identity (``broad_sample``) and optionally shared
target genes.
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


def same_perturbation_mask(
    broad_samples: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Boolean ``(B, B)`` mask of wells sharing a ``broad_sample``."""
    unique = {s: idx for idx, s in enumerate(dict.fromkeys(broad_samples))}
    ids = torch.tensor([unique[s] for s in broad_samples], device=device)
    return ids.unsqueeze(0) == ids.unsqueeze(1)


def _gene_overlap_matrix(
    target_keys: list[str],
    target_weight: float,
    device: torch.device,
) -> torch.Tensor:
    """``(B, B)`` matrix holding *target_weight* where gene sets intersect."""
    gene_ids: dict[str, int] = {}
    membership_rows: list[list[int]] = []
    for key in target_keys:
        row = [gene_ids.setdefault(gene, len(gene_ids)) for gene in parse_target_gene_key(key)]
        membership_rows.append(row)

    n_rows, n_genes = len(target_keys), len(gene_ids)
    membership = torch.zeros(n_rows, max(n_genes, 1))
    for row_idx, gene_indices in enumerate(membership_rows):
        for gene_idx in gene_indices:
            membership[row_idx, gene_idx] = 1.0

    membership = membership.to(device)
    return (membership @ membership.t() > 0).float() * target_weight


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
    affinity = same_perturbation_mask(broad_samples, device).float()

    if target_keys is not None and target_weight > 0.0:
        if len(target_keys) != len(broad_samples):
            raise ValueError("target_keys must have the same length as broad_samples")
        overlap = _gene_overlap_matrix(target_keys, target_weight, device)
        affinity = torch.maximum(affinity, overlap)

    affinity.fill_diagonal_(1.0)
    return affinity


def _row_normalize(affinity: torch.Tensor) -> torch.Tensor:
    """Normalize each row of an affinity matrix to sum to 1."""
    row_sums = affinity.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return affinity / row_sums


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

    Instead of hard diagonal targets, uses a soft label matrix where wells
    sharing a perturbation (and optionally a target gene) are positive pairs.

    Each direction normalizes its own target rows: i2t from ``A``, t2i from
    ``A.t()``. Transposing a single row-normalized matrix would leave t2i rows
    that do not sum to 1 once weights are continuous.

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

    Wells sharing a ``broad_sample`` are positives for each other. The
    self-similarity diagonal is excluded from numerator and denominator, and
    rows with no replicate in the batch are dropped from the mean. The
    similarity matrix is symmetric, so one direction suffices.

    Args:
        image_features: ``(B, D)`` L2-normalized image embeddings.
        scale: Linear logit scale: ``logit_scale.exp()`` for the shared
            learnable temperature, or ``1 / temperature`` for a fixed one.
        broad_samples: Perturbation ID per sample (length B).

    Returns:
        Scalar loss. When no row has a replicate, returns a zero still attached
        to the autograd graph so DDP gradients stay in sync.
    """
    device = image_features.device
    n = image_features.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=device)

    positives = same_perturbation_mask(broad_samples, device) & ~eye
    positive_counts = positives.sum(dim=1)
    valid = (positive_counts > 0).to(image_features.dtype)

    logits = scale * image_features @ image_features.t()
    logits = logits.masked_fill(eye, float("-inf"))
    log_probs = F.log_softmax(logits, dim=1).masked_fill(eye, 0.0)

    weights = positives.float() / positive_counts.clamp(min=1).unsqueeze(1)
    per_row = -(weights * log_probs).sum(dim=1)
    # Replicate-less rows have an all-zero weight row, so they contribute 0 to
    # the sum. Masking instead of indexing keeps this sync-free.
    return (per_row * valid).sum() / valid.sum().clamp(min=1)


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


def replicate_loss_scale(
    logit_scale: torch.Tensor,
    replicate_temperature: float | None,
) -> torch.Tensor | float:
    """Resolve the linear scale for the replicate image-image term.

    Defaults to the shared learnable temperature; *replicate_temperature* pins
    it to ``1 / temperature`` instead.

    Args:
        logit_scale: Scalar log-temperature parameter.
        replicate_temperature: Fixed temperature, or ``None`` to share.
    """
    if replicate_temperature is None:
        return logit_scale.exp()
    return 1.0 / replicate_temperature


def compute_training_loss(
    loss_type: str,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    broad_samples: list[str] | None = None,
    target_keys: list[str] | None = None,
    target_weight: float = 0.0,
    replicate_weight: float = 0.0,
    replicate_temperature: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total training loss and its named components.

    Composes the text-alignment loss with any auxiliary terms. Evaluation
    calls :func:`compute_loss` directly so ``eval_loss`` stays text-only and
    comparable across ablations.

    Args:
        loss_type: ``"infonce"`` or ``"cwcl"``.
        image_features: ``(B, D)`` L2-normalized image embeddings.
        text_features: ``(B, D)`` L2-normalized text embeddings.
        logit_scale: Scalar log-temperature parameter.
        broad_samples: Perturbation ID per sample; required for CWCL and for
            the replicate term.
        target_keys: Optional canonical gene keys per sample.
        target_weight: Affinity for gene-overlapping pairs.
        replicate_weight: Weight of the replicate image-image term (0 disables).
        replicate_temperature: Fixed temperature for the replicate term.

    Returns:
        ``(total, components)``. *components* holds the detached per-term
        values for logging, and is empty when the total is a single term.

    Raises:
        ValueError: If the replicate term is enabled without *broad_samples*.
    """
    text = compute_loss(
        loss_type,
        image_features,
        text_features,
        logit_scale,
        broad_samples=broad_samples,
        target_keys=target_keys,
        target_weight=target_weight,
    )
    if replicate_weight <= 0.0:
        return text, {}

    if broad_samples is None:
        raise ValueError("The replicate image-image term requires broad_samples")
    replicate = replicate_image_loss(
        image_features,
        replicate_loss_scale(logit_scale, replicate_temperature),
        broad_samples=broad_samples,
    )
    total = text + replicate_weight * replicate
    return total, {"text": text.detach(), "replicate": replicate.detach()}
