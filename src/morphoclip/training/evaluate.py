"""Evaluation loop and retrieval metrics for MorphoCLIP training."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from morphoclip.data.perturbation import PerturbationInfo
from morphoclip.training.batch_correction import cross_well_alignment
from morphoclip.training.engine import autocast_context
from morphoclip.training.losses import compute_loss


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


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    broad_samples: list[str] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics for both directions.

    When ``broad_samples`` is provided, a retrieval is correct if any
    sample sharing the same perturbation ID is in the top-k (handles
    duplicate text embeddings from replicate wells).  Falls back to
    diagonal-only targets when ``broad_samples`` is ``None``.

    Args:
        image_features: ``(N, D)`` image embeddings.
        text_features: ``(N, D)`` text embeddings.
        broad_samples: Perturbation ID per sample (length N).

    Returns:
        Dict with R@1/5/10, mean/median rank for both directions.
    """
    logits = image_features @ text_features.t()
    N = logits.shape[0]
    device = logits.device

    # Build positive mask: (N, N) where mask[i,j] = True if i and j
    # share the same perturbation (or just diagonal if no broad_samples)
    if broad_samples is not None:
        unique = {s: idx for idx, s in enumerate(dict.fromkeys(broad_samples))}
        ids = torch.tensor([unique[s] for s in broad_samples], device=device)
        positive_mask = ids.unsqueeze(0) == ids.unsqueeze(1)
    else:
        positive_mask = torch.eye(N, dtype=torch.bool, device=device)

    results: dict[str, float] = {}
    for prefix, score_matrix, mask in (
        ("image_to_text", logits, positive_mask),
        ("text_to_image", logits.t(), positive_mask.t()),
    ):
        order = torch.argsort(score_matrix, dim=1, descending=True)
        # For each query, find the rank of the first correct match
        ordered_mask = mask.gather(1, order)  # (N, N) reordered by score
        # argmax on bool finds first True
        best_rank = torch.argmax(ordered_mask.to(torch.int64), dim=1) + 1

        results[f"{prefix}_mean_rank"] = float(best_rank.float().mean().item())
        results[f"{prefix}_median_rank"] = float(best_rank.float().median().item())
        for k in (1, 5, 10):
            results[f"{prefix}_R@{k}"] = float((best_rank <= k).float().mean().item())
    return results


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
) -> dict[str, float]:
    """Run one evaluation epoch.

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
