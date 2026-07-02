"""Loss functions for MorphoCLIP training.

Provides InfoNCE (standard symmetric CLIP loss) and CWCL (Continuously
Weighted Contrastive Loss with perturbation-identity soft labels).

Note on naming: CellCLIP uses "CWCL" for *Channel-Wise* Contrastive Loss
(soft labels from per-channel image similarity).  MorphoCLIP's CWCL is
*Continuously Weighted* — soft labels come from perturbation identity
(``broad_sample``), not channel similarity.
"""

import torch
import torch.nn.functional as F


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


def build_soft_labels(
    broad_samples: list[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build row-normalized soft label matrix from perturbation identity.

    Samples sharing the same ``broad_sample`` get equal positive weight;
    unrelated samples get zero.

    Args:
        broad_samples: List of perturbation IDs (length B).
        device: Target device.

    Returns:
        Soft label matrix ``(B, B)``, rows sum to 1.
    """
    # Map each unique broad_sample to an integer for vectorized comparison
    unique = {s: idx for idx, s in enumerate(dict.fromkeys(broad_samples))}
    ids = torch.tensor([unique[s] for s in broad_samples], device=device)
    labels = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
    # Row-normalize so each row sums to 1
    row_sums = labels.sum(dim=1, keepdim=True).clamp(min=1)
    return labels / row_sums


def cwcl_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    broad_samples: list[str],
) -> torch.Tensor:
    """Continuously Weighted Contrastive Loss with perturbation soft labels.

    Instead of hard diagonal targets, uses a soft label matrix where all
    wells sharing the same perturbation (``broad_sample``) are positive
    pairs with equal weight.

    Args:
        image_features: ``(B, D)`` L2-normalized image embeddings.
        text_features: ``(B, D)`` L2-normalized text embeddings.
        logit_scale: Scalar log-temperature parameter.
        broad_samples: Perturbation ID per sample (length B).

    Returns:
        Scalar loss.
    """
    logits = logit_scale.exp() * image_features @ text_features.t()
    soft_labels = build_soft_labels(broad_samples, device=logits.device)

    # Soft cross-entropy: -sum(W * log_softmax(S)) per row, averaged
    log_probs_i2t = F.log_softmax(logits, dim=1)
    log_probs_t2i = F.log_softmax(logits.t(), dim=1)

    loss_i2t = -(soft_labels * log_probs_i2t).sum(dim=1).mean()
    loss_t2i = -(soft_labels.t() * log_probs_t2i).sum(dim=1).mean()

    return (loss_i2t + loss_t2i) / 2


def compute_loss(
    loss_type: str,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    broad_samples: list[str] | None = None,
) -> torch.Tensor:
    """Dispatch to the appropriate loss function.

    Args:
        loss_type: ``"infonce"`` or ``"cwcl"``.
        image_features: ``(B, D)`` L2-normalized image embeddings.
        text_features: ``(B, D)`` L2-normalized text embeddings.
        logit_scale: Scalar log-temperature parameter.
        broad_samples: Required for CWCL — perturbation ID per sample.

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
        )
    raise ValueError(f"Unknown loss_type: {loss_type!r}")
