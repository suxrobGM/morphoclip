"""Checkpointing and the training step for MorphoCLIP."""

from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from morphoclip.training.batch_correction import PlateOffsets
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import LogitScaleModule
from morphoclip.training.metrics import compute_grad_norm
from morphoclip.training.optim import unwrap, unwrap_state_dict
from morphoclip.utils.device import autocast_context


def _get_logit_scale_data(logit_scale: nn.Parameter | nn.Module) -> torch.Tensor:
    """Extract the raw logit_scale tensor from Parameter or LogitScaleModule."""
    if isinstance(logit_scale, LogitScaleModule):
        return logit_scale.scale.data
    if hasattr(logit_scale, "module"):
        # DDP-wrapped LogitScaleModule
        return cast(LogitScaleModule, logit_scale.module).scale.data
    return logit_scale.data


def save_checkpoint(
    path: Path,
    *,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    logit_scale: nn.Parameter | nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_eval_loss: float,
    config: MorphoCLIPTrainingConfig,
    plate_offsets: PlateOffsets | None = None,
) -> None:
    """Save a training checkpoint.

    Automatically unwraps DDP wrappers so checkpoints are portable
    between single-GPU and multi-GPU modes.

    ``plate_offsets`` is stored so eval, inference and profile export apply the
    same CWA correction the run was trained under. It is ``None`` when CWA is
    off, and absent entirely from checkpoints written before offsets existed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_encoder": unwrap_state_dict(image_encoder),
            "text_projection": unwrap_state_dict(text_projection),
            "logit_scale": _get_logit_scale_data(logit_scale),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "steps": global_step,
            "best_eval_loss": best_eval_loss,
            "config": config.to_dict(),
            "plate_offsets": plate_offsets.state_dict() if plate_offsets is not None else None,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    logit_scale: nn.Parameter | nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int, float]:
    """Load a training checkpoint and restore all state.

    Handles both raw ``nn.Parameter`` and ``LogitScaleModule``
    for the logit_scale argument.

    Args:
        path: Path to the checkpoint file.
        image_encoder: Image encoder to restore weights into.
        text_projection: Text projection head to restore weights into.
        logit_scale: Learnable temperature parameter to restore.
        optimizer: Optimizer to restore state into.
        scheduler: LR scheduler to restore state into.
        device: Target device for loading.

    Returns:
        ``(start_epoch, global_step, best_eval_loss)`` from the checkpoint.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Unwrap DDP if present on the target modules
    img_target = unwrap(image_encoder)
    txt_target = unwrap(text_projection)

    img_target.load_state_dict(ckpt["image_encoder"])
    txt_target.load_state_dict(ckpt["text_projection"])

    # Handle LogitScaleModule or raw nn.Parameter
    scale_data = ckpt["logit_scale"]
    if isinstance(logit_scale, LogitScaleModule):
        logit_scale.scale.data.copy_(scale_data)
    elif hasattr(logit_scale, "module"):
        cast(LogitScaleModule, logit_scale.module).scale.data.copy_(scale_data)
    else:
        logit_scale.data.copy_(scale_data)

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["steps"], ckpt["best_eval_loss"]


def scale_param(logit_scale: nn.Module) -> nn.Parameter:
    """Get the raw scale parameter, unwrapping DDP if needed."""
    inner = cast(LogitScaleModule, unwrap(logit_scale))
    return inner.scale


def forward_step(
    batch: dict,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    text_cache: dict,
    *,
    device: torch.device,
    amp: bool,
    plate_offsets: PlateOffsets | None,
    use_ddp: bool,
    dist_state: Any,
    target_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str] | None]:
    """Forward pass + optional CWA + gather across GPUs.

    Returns ``(all_image, all_text, image_emb, text_emb, all_broad_samples,
    all_target_keys)``. The target keys feed the gene-aware CWCL soft labels and
    are ``None`` unless *target_weight* is positive, which skips both the
    per-well gene parsing and an extra all-gather.
    """
    # Lazy imports to avoid circular dependency (engine <-> evaluate)
    from morphoclip.data.perturbation import target_gene_key
    from morphoclip.training.distributed import all_gather_tensors, gather_string_lists
    from morphoclip.training.evaluate import lookup_text_embeddings

    features = batch["features"].to(device, non_blocking=True)
    site_mask = batch["site_mask"].to(device, non_blocking=True)
    pert_infos = batch["pert_info"]

    with autocast_context(device, amp):
        image_emb = image_encoder(features, site_mask)
        raw_text = lookup_text_embeddings(pert_infos, text_cache, device)
        text_emb = text_projection(raw_text)

        if plate_offsets is not None:
            image_emb = plate_offsets.apply(image_emb, batch["plates"])

        broad_samples = [info.broad_sample for info in pert_infos]
        target_keys = (
            [target_gene_key(info) for info in pert_infos] if target_weight > 0.0 else None
        )
        if use_ddp:
            # with_grad keeps gradients flowing through all_gather, so negatives
            # on remote GPUs still contribute to the contrastive loss.
            all_image = all_gather_tensors(image_emb, with_grad=True)
            all_text = all_gather_tensors(text_emb, with_grad=True)
            all_broad = gather_string_lists(broad_samples, dist_state.world_size)
            all_targets = (
                gather_string_lists(target_keys, dist_state.world_size)
                if target_keys is not None
                else None
            )
        else:
            all_image, all_text = image_emb, text_emb
            all_broad, all_targets = broad_samples, target_keys

    return all_image, all_text, image_emb, text_emb, all_broad, all_targets


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    grad_scaler: torch.amp.GradScaler,
    all_params: list[nn.Parameter],
    logit_scale: nn.Module,
    *,
    grad_clip_norm: float,
    logit_scale_max: float,
) -> tuple[float, float]:
    """Unscale, clip, step, clamp scale. Returns (grad_norm_before, grad_norm_after)."""
    grad_scaler.unscale_(optimizer)
    grad_norm_before = float(nn.utils.clip_grad_norm_(all_params, grad_clip_norm))
    grad_norm_after = compute_grad_norm(all_params)
    grad_scaler.step(optimizer)
    grad_scaler.update()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        scale_param(logit_scale).data.clamp_(0, logit_scale_max)

    return grad_norm_before, grad_norm_after
