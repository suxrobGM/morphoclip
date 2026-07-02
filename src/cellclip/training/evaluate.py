"""Evaluation epoch and retrieval metrics for local CellCLIP training."""

from typing import cast

import torch
from torch import nn

from cellclip.training.losses import compute_loss
from cellclip.training.model import CellCLIP
from morphoclip.utils.device import autocast_context


def _move_tokens(tokens: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in tokens.items()}


def _move_optional_tokens(
    tokens: dict[str, torch.Tensor] | None,
    device: torch.device,
) -> dict[str, torch.Tensor] | None:
    if tokens is None:
        return None
    return _move_tokens(tokens, device)


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> dict[str, float]:
    """Compute retrieval metrics for both directions."""
    logits = image_features @ text_features.t()
    results: dict[str, float] = {}
    for prefix, score_matrix in (
        ("image_to_text", logits),
        ("text_to_image", logits.t()),
    ):
        order = torch.argsort(score_matrix, dim=1, descending=True)
        target = torch.arange(score_matrix.shape[0], device=score_matrix.device).unsqueeze(1)
        ranks = torch.argmax((order == target).to(torch.int64), dim=1) + 1
        results[f"{prefix}_mean_rank"] = float(ranks.float().mean().item())
        results[f"{prefix}_median_rank"] = float(ranks.float().median().item())
        for k in (1, 5, 10):
            results[f"{prefix}_R@{k}"] = float((ranks <= k).float().mean().item())
    return results


def evaluate_epoch(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    loss_type: str,
    amp: bool,
) -> dict[str, float]:
    """Run one evaluation epoch."""
    model.eval()
    losses: list[float] = []
    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []

    # Unwrap DDP for encode_mil access
    raw_model = cast(CellCLIP, model.module if hasattr(model, "module") else model)

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            text_tokens = _move_tokens(batch["text_tokens"], device)
            smiles_tokens = _move_optional_tokens(batch.get("smiles_tokens"), device)
            has_smiles = batch.get("has_smiles")
            if has_smiles is not None:
                has_smiles = has_smiles.to(device, non_blocking=True)
            with autocast_context(device, amp):
                pooled_images = raw_model.encode_mil(features)
                outputs = model(
                    pooled_images,
                    text_tokens,
                    smiles=smiles_tokens,
                    has_smiles=has_smiles,
                )
                image_features, text_features, logit_scale = outputs[:3]
                loss = compute_loss(
                    loss_type,
                    pooled_images,
                    image_features,
                    text_features,
                    logit_scale,
                )
            losses.append(float(loss.detach().cpu().item()))
            image_batches.append(image_features.detach().cpu())
            text_batches.append(text_features.detach().cpu())

    metrics = {"eval_loss": float(sum(losses) / max(1, len(losses)))}
    if image_batches:
        metrics.update(
            compute_retrieval_metrics(
                torch.cat(image_batches, dim=0),
                torch.cat(text_batches, dim=0),
            )
        )
    return metrics
