"""Optimizer, learning-rate schedule, and DDP unwrapping.

Shared by both trainers. CellCLIP imports this rather than carrying its own
copy, which is why :func:`split_params` takes *decay_first*: the two packages
order their parameter groups differently and optimizer state is keyed by group
index, so a resumed run would map momentum onto the wrong tensors if the order
changed.
"""

import math
from typing import Any, cast

from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from morphoclip.training.config import ADAMW_BETAS, ADAMW_EPS, MorphoCLIPTrainingConfig


def split_params(
    *modules: nn.Module,
    weight_decay: float,
    decay_first: bool = True,
) -> list[dict[str, Any]]:
    """Split trainable parameters into decay and no-decay groups.

    Biases, LayerNorm weights, and any parameter with fewer than two dimensions
    get zero weight decay. That last rule already covers the 0-d learnable
    temperature.

    Args:
        modules: Modules whose parameters to split.
        weight_decay: Decay applied to the decay group.
        decay_first: Put the decay group first. CellCLIP passes ``False`` to keep
            the group order its checkpoints were written with.

    Returns:
        Two AdamW param-group dicts, ordered per *decay_first*.
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

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return groups if decay_first else groups[::-1]


def build_optimizer(
    params: list[dict[str, Any]],
    config: MorphoCLIPTrainingConfig,
) -> AdamW:
    """Build the MorphoCLIP AdamW optimizer from config."""
    return AdamW(params, lr=config.optimization.lr, betas=ADAMW_BETAS, eps=ADAMW_EPS)


def build_warmup_cosine_scheduler(
    optimizer: AdamW,
    *,
    total_steps: int,
    warmup_steps: int,
) -> LambdaLR:
    """Linear warmup over *warmup_steps*, then cosine decay to zero at *total_steps*."""

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def unwrap(module: nn.Module) -> nn.Module:
    """Return the inner module, stripping a DDP wrapper if present."""
    if hasattr(module, "module"):
        return cast(nn.Module, module.module)
    return module


def unwrap_state_dict(module: nn.Module) -> dict[str, Any]:
    """State dict with any DDP wrapper stripped, so checkpoints stay portable."""
    return unwrap(module).state_dict()
