"""Optimizer, scheduler, checkpointing, and training step utilities."""

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import LogitScaleModule
from morphoclip.training.metrics import compute_grad_norm
from morphoclip.utils.device import autocast_context, resolve_device  # noqa: F401


def split_params(
    *modules: nn.Module,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups.

    Biases, LayerNorm weights, and 1-d params get zero weight decay.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            lowered = name.lower()
            if param.ndim < 2 or "bias" in lowered or "ln" in lowered or "norm" in lowered:
                no_decay.append(param)
            else:
                decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(
    params: list[dict[str, Any]],
    config: MorphoCLIPTrainingConfig,
) -> AdamW:
    """Build AdamW optimizer from config."""
    opt = config.optimization
    return AdamW(params, lr=opt.lr, betas=opt.betas, eps=opt.eps)


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Warmup + cosine decay learning-rate schedule."""

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _unwrap_state_dict(module: nn.Module) -> dict[str, Any]:
    """Get state_dict from a module, stripping DDP wrapper if present."""
    inner = module.module if hasattr(module, "module") else module
    return inner.state_dict()


def _get_logit_scale_data(logit_scale: nn.Parameter | nn.Module) -> torch.Tensor:
    """Extract the raw logit_scale tensor from Parameter or LogitScaleModule."""
    if isinstance(logit_scale, LogitScaleModule):
        return logit_scale.scale.data
    if hasattr(logit_scale, "module"):
        # DDP-wrapped LogitScaleModule
        return logit_scale.module.scale.data
    return logit_scale.data


def save_checkpoint(
    path: Path,
    *,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    logit_scale: nn.Parameter | nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_eval_loss: float,
    config: MorphoCLIPTrainingConfig,
) -> None:
    """Save a training checkpoint.

    Automatically unwraps DDP wrappers so checkpoints are portable
    between single-GPU and multi-GPU modes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_encoder": _unwrap_state_dict(image_encoder),
            "text_projection": _unwrap_state_dict(text_projection),
            "logit_scale": _get_logit_scale_data(logit_scale),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "steps": global_step,
            "best_eval_loss": best_eval_loss,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    logit_scale: nn.Parameter | nn.Module,
    optimizer: AdamW,
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
    img_target = image_encoder.module if hasattr(image_encoder, "module") else image_encoder
    txt_target = text_projection.module if hasattr(text_projection, "module") else text_projection

    img_target.load_state_dict(ckpt["image_encoder"])
    txt_target.load_state_dict(ckpt["text_projection"])

    # Handle LogitScaleModule or raw nn.Parameter
    scale_data = ckpt["logit_scale"]
    if isinstance(logit_scale, LogitScaleModule):
        logit_scale.scale.data.copy_(scale_data)
    elif hasattr(logit_scale, "module"):
        logit_scale.module.scale.data.copy_(scale_data)
    else:
        logit_scale.data.copy_(scale_data)

    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["steps"], ckpt["best_eval_loss"]


# --- Per-step helpers ---


def scale_param(logit_scale: nn.Module) -> nn.Parameter:
    """Get the raw scale parameter, unwrapping DDP if needed."""
    return logit_scale.module.scale if hasattr(logit_scale, "module") else logit_scale.scale


def forward_step(
    batch: dict,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    text_cache: dict,
    *,
    device: torch.device,
    amp: bool,
    use_cwa: bool,
    use_ddp: bool,
    dist_cfg: Any,
    dist_state: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Forward pass + optional CWA + gather across GPUs.

    Returns (all_image, all_text, image_emb, text_emb, all_broad_samples).
    """
    # Lazy imports to avoid circular dependency (engine <-> evaluate)
    from morphoclip.training.batch_correction import cross_well_alignment
    from morphoclip.training.distributed import all_gather_tensors, gather_string_lists
    from morphoclip.training.evaluate import lookup_text_embeddings

    features = batch["features"].to(device, non_blocking=True)
    site_mask = batch["site_mask"].to(device, non_blocking=True)
    pert_infos = batch["pert_info"]

    with autocast_context(device, amp):
        image_emb = image_encoder(features, site_mask)
        raw_text = lookup_text_embeddings(pert_infos, text_cache, device)
        text_emb = text_projection(raw_text)

        if use_cwa:
            image_emb = cross_well_alignment(image_emb, batch["plates"])

        broad_samples = [info.broad_sample for info in pert_infos]
        if use_ddp:
            all_image = all_gather_tensors(image_emb, with_grad=dist_cfg.gather_with_grad)
            all_text = all_gather_tensors(text_emb, with_grad=dist_cfg.gather_with_grad)
            all_broad = gather_string_lists(broad_samples, dist_state.world_size)
        else:
            all_image, all_text, all_broad = image_emb, text_emb, broad_samples

    return all_image, all_text, image_emb, text_emb, all_broad


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
