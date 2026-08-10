"""Optimizer construction and checkpointing for local CellCLIP training."""

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from cellclip.training.config import CellCLIPTrainingConfig
from morphoclip.training.optim import split_params, unwrap_state_dict


def build_optimizer(model: nn.Module, config: CellCLIPTrainingConfig) -> AdamW:
    """Build AdamW with CLIP-style decay exclusions.

    The no-decay group goes first, matching the order the published run's
    checkpoints were written with.
    """
    opt = config.optimization
    return AdamW(
        split_params(model, weight_decay=opt.weight_decay, decay_first=False),
        lr=opt.lr,
        betas=opt.betas,
        eps=opt.eps,
    )


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
            "model": unwrap_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "steps": global_step,
            "best_eval_loss": best_eval_loss,
            "config": config.to_dict(),
        },
        path,
    )
