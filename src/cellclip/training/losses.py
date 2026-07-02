"""Loss functions for local CellCLIP training."""

import torch
from torch import nn


def clip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    loss_fct_img: nn.Module | None = None,
    loss_fct_txt: nn.Module | None = None,
) -> torch.Tensor:
    """Symmetric CLIP loss."""
    if loss_fct_img is None:
        loss_fct_img = nn.CrossEntropyLoss()
    if loss_fct_txt is None:
        loss_fct_txt = nn.CrossEntropyLoss()

    logits_per_image = logit_scale.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    ground_truth = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_img = loss_fct_img(logits_per_image, ground_truth) / 2
    loss_txt = loss_fct_txt(logits_per_text, ground_truth) / 2
    return loss_img + loss_txt


def cwcl_loss(
    pooled_images: torch.Tensor,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    loss_fct_txt: nn.Module | None = None,
) -> torch.Tensor:
    """Channel-wise soft-label CellCLIP loss."""
    if loss_fct_txt is None:
        loss_fct_txt = nn.CrossEntropyLoss()

    logits_per_image = logit_scale.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    ground_truth = torch.arange(len(logits_per_image), device=pooled_images.device)
    loss_cl = loss_fct_txt(logits_per_text, ground_truth)

    batch_size, num_channels, _ = pooled_images.shape
    images_norm = pooled_images / (pooled_images.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = torch.zeros(batch_size, batch_size, device=pooled_images.device)
    for channel_idx in range(num_channels):
        channel_features = images_norm[:, channel_idx, :]
        sim_matrix += channel_features @ channel_features.t()

    sim_matrix = sim_matrix / (num_channels * 2) + 0.5
    sim_matrix = sim_matrix / (sim_matrix.sum(dim=1, keepdim=True) + 1e-8)
    loss_cwcl = -torch.sum(sim_matrix * torch.log_softmax(logits_per_image, dim=1)) / batch_size
    return loss_cl + loss_cwcl


def compute_loss(
    loss_type: str,
    pooled_images: torch.Tensor,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Dispatch local loss implementations."""
    normalized = str(loss_type).strip().lower()
    if normalized == "clip":
        return clip_loss(image_features, text_features, logit_scale)
    if normalized == "cwcl":
        return cwcl_loss(pooled_images, image_features, text_features, logit_scale)
    raise ValueError(f"Unsupported loss_type: {loss_type!r}")
