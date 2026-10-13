"""Evaluation loop and retrieval metrics for MorphoCLIP training."""

from collections.abc import Iterable

import torch
from torch import nn

from morphoclip.data.perturbation import PerturbationInfo, target_gene_key
from morphoclip.training.batch_correction import PlateOffsets
from morphoclip.training.forward import encode_batch
from morphoclip.training.losses import compute_loss
from morphoclip.training.retrieval import compute_retrieval_metrics
from morphoclip.utils.device import autocast_context


def evaluate_epoch(
    image_encoder: nn.Module,
    text_projection: nn.Module,
    text_cache: dict,
    loader: Iterable[dict],
    *,
    device: torch.device,
    logit_scale: nn.Parameter,
    loss_type: str,
    amp: bool,
    plate_offsets: PlateOffsets | None = None,
    target_weight: float = 0.0,
) -> dict[str, float]:
    """Run one evaluation epoch.

    The eval loss covers text alignment only. The replicate image-image term is
    left out so model selection stays comparable across ablations.

    Args:
        image_encoder: Encoder producing one embedding per well.
        text_projection: Projection head over cached BERT features.
        text_cache: Cached text features.
        loader: DataLoader over the evaluation wells.
        device: Device to run on.
        logit_scale: Learnable temperature.
        loss_type: Which contrastive loss to score with.
        amp: Run the forward pass under autocast.
        plate_offsets: CWA offsets to subtract, or ``None`` to evaluate
            uncorrected embeddings.
        target_weight: Gene-aware CWCL affinity weight.

    Returns:
        Dict with ``eval_loss`` and retrieval metrics.
    """
    image_encoder.eval()
    text_projection.eval()

    losses: list[float] = []
    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []
    all_broad_samples: list[str] = []

    with torch.no_grad():
        for batch in loader:
            pert_infos: list[PerturbationInfo] = batch["pert_info"]

            with autocast_context(device, amp):
                image_emb, text_emb = encode_batch(
                    batch,
                    image_encoder,
                    text_projection,
                    text_cache,
                    device=device,
                    plate_offsets=plate_offsets,
                )

                broad_samples = [info.broad_sample for info in pert_infos]
                loss = compute_loss(
                    loss_type,
                    image_emb,
                    text_emb,
                    logit_scale,
                    broad_samples=broad_samples,
                    target_keys=(
                        [target_gene_key(info) for info in pert_infos]
                        if target_weight > 0.0
                        else None
                    ),
                    target_weight=target_weight,
                )

            losses.append(float(loss.detach().cpu().item()))
            image_batches.append(image_emb.detach().cpu())
            text_batches.append(text_emb.detach().cpu())
            all_broad_samples.extend(broad_samples)

    metrics: dict[str, float] = {
        "eval_loss": float(sum(losses) / max(1, len(losses))),
    }
    if image_batches:
        metrics.update(
            compute_retrieval_metrics(
                torch.cat(image_batches, dim=0),
                torch.cat(text_batches, dim=0),
                broad_samples=all_broad_samples,
            )
        )
    return metrics
