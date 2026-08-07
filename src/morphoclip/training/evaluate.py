"""Evaluation loop and retrieval metrics for MorphoCLIP training."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from morphoclip.data.perturbation import PerturbationInfo, target_gene_key
from morphoclip.training.batch_correction import cross_well_alignment
from morphoclip.training.engine import autocast_context
from morphoclip.training.losses import compute_loss
from morphoclip.training.retrieval import compute_retrieval_metrics

__all__ = ["compute_retrieval_metrics", "evaluate_epoch", "lookup_text_embeddings"]


def lookup_text_embeddings(
    pert_infos: list[PerturbationInfo],
    text_cache: dict,
    device: torch.device,
) -> torch.Tensor:
    """Look up cached 768-d BERT features for a batch of perturbations.

    Args:
        pert_infos: PerturbationInfo for each sample in the batch.
        text_cache: Dict from ``load_cached_text_features`` with
            ``embeddings`` and ``id_to_idx``.
        device: Target device.

    Returns:
        Raw BERT features ``(B, 768)`` on *device*.
    """
    id_to_idx = text_cache["id_to_idx"]
    embeddings = text_cache["embeddings"]
    indices = [id_to_idx[info.broad_sample] for info in pert_infos]
    return embeddings[indices].to(device, non_blocking=True)


def evaluate_epoch(
    image_encoder: nn.Module,
    text_projection: nn.Module,
    text_cache: dict,
    loader: DataLoader,
    *,
    device: torch.device,
    logit_scale: nn.Parameter,
    loss_type: str,
    use_cwa: bool,
    amp: bool,
    target_weight: float = 0.0,
) -> dict[str, float]:
    """Run one evaluation epoch.

    The eval loss covers text alignment only. The replicate image-image term is
    left out so model selection stays comparable across ablations.

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
            features = batch["features"].to(device, non_blocking=True)
            site_mask = batch["site_mask"].to(device, non_blocking=True)
            pert_infos: list[PerturbationInfo] = batch["pert_info"]

            with autocast_context(device, amp):
                image_emb = image_encoder(features, site_mask)
                raw_text = lookup_text_embeddings(pert_infos, text_cache, device)
                text_emb = text_projection(raw_text)

                if use_cwa:
                    image_emb = cross_well_alignment(image_emb, batch["plates"])

                broad_samples = [info.broad_sample for info in pert_infos]
                loss = compute_loss(
                    loss_type,
                    image_emb,
                    text_emb,
                    logit_scale,
                    broad_samples=broad_samples,
                    target_keys=[target_gene_key(info) for info in pert_infos],
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
