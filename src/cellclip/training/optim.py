"""Optimizer, scheduler, and checkpointing for local CellCLIP training."""

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from cellclip.training.config import CellCLIPTrainingConfig


def build_optimizer(model: nn.Module, config: CellCLIPTrainingConfig) -> AdamW:
    """Build AdamW with CLIP-style decay exclusions (logit_scale excluded)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if (
            param.ndim < 2
            or "bias" in lowered
            or "ln" in lowered
            or "norm" in lowered
            or "logit_scale" in lowered
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    opt_cfg = config.optimization
    return AdamW(
        [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": opt_cfg.weight_decay},
        ],
        lr=opt_cfg.lr,
        betas=opt_cfg.betas,
        eps=opt_cfg.eps,
    )


def build_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Warmup + cosine decay scheduler."""

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _unwrap_state_dict(module: nn.Module) -> dict[str, Any]:
    """Get state_dict stripping DDP wrapper if present."""
    inner = cast(nn.Module, module.module) if hasattr(module, "module") else module
    return inner.state_dict()


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_eval_loss: float,
    config: CellCLIPTrainingConfig,
) -> None:
    """Save a training checkpoint (DDP-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _unwrap_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "steps": global_step,
            "best_eval_loss": best_eval_loss,
            "config": asdict(config),
        },
        path,
    )
