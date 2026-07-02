"""Model, optimizer, and console-banner setup for MorphoCLIP training."""

import math
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import DistributedState, LogitScaleModule
from morphoclip.training.engine import build_optimizer, build_scheduler, split_params
from morphoclip.training.inference import build_models
from morphoclip.utils.device import build_grad_scaler

# OpenAI CLIP initial temperature (tau=0.07 -> logit_scale = ln(1/0.07))
CLIP_INIT_LOGIT_SCALE = float(np.log(1.0 / 0.07))


def log_device_banner(
    console: Console,
    device: torch.device,
    *,
    use_ddp: bool,
    world_size: int,
) -> None:
    """Print the device banner (GPU/MPS/CPU) at training start."""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        console.print(f"Device: [bold green]{gpu_name}[/bold green] ({vram:.1f} GB)")
        if use_ddp:
            console.print(f"DDP: [bold green]{world_size} GPUs[/bold green]")
    elif device.type == "mps":
        console.print("Device: [bold green]Apple Silicon (MPS)[/bold green]")
    else:
        console.print("Device: [bold yellow]CPU[/bold yellow]")


def log_config_summary(
    console: Console,
    config: MorphoCLIPTrainingConfig,
    image_encoder: nn.Module,
    text_projection: nn.Module,
) -> None:
    """Print config summary and trainable parameter counts."""
    table = Table(title="MorphoCLIP Training Config", show_lines=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Loss", config.optimization.loss_type)
    table.add_row("CWA", str(config.optimization.use_cwa))
    table.add_row("Learning rate", f"{config.optimization.lr:.1e}")
    table.add_row("Weight decay", f"{config.optimization.weight_decay}")
    table.add_row("Epochs", str(config.optimization.epochs))
    table.add_row("Batch size", str(config.dataset.batch_size))
    agg = config.model.channel_aggregation
    agg_label = f"ccf ({config.model.ccf_layers}L)" if agg == "ccf" else agg
    table.add_row("Channel agg", agg_label)
    table.add_row("Output dim", str(config.model.output_dim))
    table.add_row("AMP", str(config.runtime.amp))
    table.add_row("Device", config.runtime.device)
    if config.distributed.enabled:
        table.add_row("DDP", "enabled")
        table.add_row("Grad accum", str(config.distributed.gradient_accumulation_steps))

    console.print(table)

    img_params = sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)
    txt_params = sum(p.numel() for p in text_projection.parameters() if p.requires_grad)
    console.print(
        f"  Trainable params: [bold]{img_params + txt_params + 1:,}[/bold] "
        f"(image encoder: {img_params:,}, text projection: {txt_params:,})"
    )


def build_and_wrap_models(
    config: MorphoCLIPTrainingConfig,
    device: torch.device,
    *,
    use_ddp: bool,
    dist_state: DistributedState | None,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Build the image/text/logit-scale modules, wrapping in DDP when requested."""
    image_encoder: nn.Module
    text_projection: nn.Module
    image_encoder, text_projection = build_models(config, device)
    logit_scale: nn.Module = LogitScaleModule(init_value=CLIP_INIT_LOGIT_SCALE, device=device)

    if use_ddp:
        assert dist_state is not None  # implied by use_ddp
        device_ids = [dist_state.local_rank]
        find_unused = config.distributed.find_unused_parameters
        image_encoder = DDP(
            image_encoder, device_ids=device_ids, find_unused_parameters=find_unused
        )
        text_projection = DDP(
            text_projection, device_ids=device_ids, find_unused_parameters=find_unused
        )
        logit_scale = DDP(logit_scale, device_ids=device_ids, find_unused_parameters=find_unused)

    return image_encoder, text_projection, logit_scale


def build_optimization(
    image_encoder: nn.Module,
    text_projection: nn.Module,
    logit_scale: nn.Module,
    config: MorphoCLIPTrainingConfig,
    *,
    device: torch.device,
    num_batches: int,
) -> tuple[AdamW, LambdaLR, Any, int]:
    """Build optimizer, scheduler, grad scaler, and resolve total_steps.

    logit_scale parameters get their own no-decay group after the split param
    groups, matching the original ordering.
    """
    opt_cfg = config.optimization
    accum_steps = config.distributed.gradient_accumulation_steps

    param_groups = split_params(image_encoder, text_projection, weight_decay=opt_cfg.weight_decay)
    param_groups.append({"params": list(logit_scale.parameters()), "weight_decay": 0.0})
    optimizer = build_optimizer(param_groups, config)

    steps_per_epoch = math.ceil(num_batches / accum_steps)
    total_steps = max(1, steps_per_epoch * opt_cfg.epochs)
    if config.runtime.max_train_steps is not None:
        total_steps = min(total_steps, config.runtime.max_train_steps)
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=opt_cfg.warmup_steps
    )
    grad_scaler = build_grad_scaler(device, enabled=config.runtime.amp)

    return optimizer, scheduler, grad_scaler, total_steps
