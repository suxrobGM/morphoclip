"""MorphoCLIP training orchestrator.

Wires together data loading, model creation, and the training loop.
Supports single-GPU and multi-GPU (DDP via ``torchrun``) training.
"""

import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as torch_dist
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.table import Table
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import (
    DistributedState,
    LogitScaleModule,
    all_reduce_scalar,
    cleanup_distributed,
    setup_distributed,
)
from morphoclip.training.engine import (
    autocast_context,
    build_optimizer,
    build_scheduler,
    forward_step,
    load_checkpoint,
    optimizer_step,
    resolve_device,
    save_checkpoint,
    scale_param,
    split_params,
)
from morphoclip.training.evaluate import evaluate_epoch
from morphoclip.training.inference import build_models
from morphoclip.training.losses import compute_loss
from morphoclip.training.metrics import compute_logit_stats
from morphoclip.training.tb_logger import TrainingLogger
from morphoclip.training.train_data import build_train_data
from morphoclip.utils.caching import load_cached_text_features
from morphoclip.utils.device import build_grad_scaler

console = Console()

# OpenAI CLIP initial temperature (tau=0.07 -> logit_scale = ln(1/0.07))
CLIP_INIT_LOGIT_SCALE = float(np.log(1.0 / 0.07))


# --- Setup helpers ---


def _log_setup(
    config: MorphoCLIPTrainingConfig, image_encoder: nn.Module, text_projection: nn.Module
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


# --- Entry point ---


def train_morphoclip(
    config: MorphoCLIPTrainingConfig,
    *,
    run_dir: Path,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    """Run MorphoCLIP training and return a summary."""
    dist_cfg = config.distributed

    dist_state: DistributedState | None = None
    if dist_cfg.enabled:
        dist_state = setup_distributed(dist_cfg.backend)
        device = dist_state.device
        is_main = dist_state.is_main
    else:
        device = resolve_device(config.runtime.device)
        is_main = True

    torch.manual_seed(config.runtime.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.runtime.seed)

    try:
        return _train_loop(
            config,
            run_dir=run_dir,
            resume_from=resume_from,
            device=device,
            dist_state=dist_state,
            is_main=is_main,
        )
    finally:
        if dist_state is not None:
            cleanup_distributed()


# --- Training loop ---


def _train_loop(
    config: MorphoCLIPTrainingConfig,
    *,
    run_dir: Path,
    resume_from: Path | None,
    device: torch.device,
    dist_state: DistributedState | None,
    is_main: bool,
) -> dict[str, Any]:
    dist_cfg = config.distributed
    opt_cfg = config.optimization
    tb_cfg = config.tensorboard
    use_ddp = dist_state is not None and dist_state.world_size > 1
    accum_steps = dist_cfg.gradient_accumulation_steps

    # --- Console ---
    if is_main:
        console.rule("[bold blue]MorphoCLIP Training")
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            vram = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            console.print(f"Device: [bold green]{gpu_name}[/bold green] ({vram:.1f} GB)")
            if use_ddp:
                console.print(f"DDP: [bold green]{dist_state.world_size} GPUs[/bold green]")
        elif device.type == "mps":
            console.print("Device: [bold green]Apple Silicon (MPS)[/bold green]")
        else:
            console.print("Device: [bold yellow]CPU[/bold yellow]")

    # --- Data ---
    if is_main:
        console.print("\n[bold]Loading data...[/bold]")
    train_loader, val_loader, train_count, val_count, train_sampler = build_train_data(
        config,
        device,
        dist_state=dist_state,
    )
    if is_main:
        console.print(f"  Train: {train_count:,} | Val: {val_count:,}")

    text_cache = load_cached_text_features(config.dataset.text_cache_path)
    if is_main:
        console.print(f"  Text cache: {text_cache['embeddings'].shape[0]:,} perturbations")

    # --- Models ---
    if is_main:
        console.print("\n[bold]Building models...[/bold]")
    image_encoder, text_projection = build_models(config, device)
    logit_scale: nn.Module = LogitScaleModule(init_value=CLIP_INIT_LOGIT_SCALE, device=device)
    if is_main:
        _log_setup(config, image_encoder, text_projection)

    if use_ddp:
        ddp_kwargs = {
            "device_ids": [dist_state.local_rank],
            "find_unused_parameters": dist_cfg.find_unused_parameters,
        }
        image_encoder = DDP(image_encoder, **ddp_kwargs)
        text_projection = DDP(text_projection, **ddp_kwargs)
        logit_scale = DDP(logit_scale, **ddp_kwargs)

    # --- Optimizer & scheduler ---
    param_groups = split_params(image_encoder, text_projection, weight_decay=opt_cfg.weight_decay)
    param_groups.append({"params": list(logit_scale.parameters()), "weight_decay": 0.0})
    optimizer = build_optimizer(param_groups, config)

    steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = max(1, steps_per_epoch * opt_cfg.epochs)
    if config.runtime.max_train_steps is not None:
        total_steps = min(total_steps, config.runtime.max_train_steps)
    scheduler = build_scheduler(
        optimizer, total_steps=total_steps, warmup_steps=opt_cfg.warmup_steps
    )
    grad_scaler = build_grad_scaler(device, enabled=config.runtime.amp)

    # --- Training state ---
    run_dir.mkdir(parents=True, exist_ok=True)
    rank = dist_state.rank if dist_state else 0
    logger = TrainingLogger(run_dir, config.tensorboard, rank=rank)
    logger.log_config(config)

    history: list[dict[str, float | int]] = []
    best_eval_loss = float("inf")
    global_step = 0
    start_epoch = 1
    best_ckpt = run_dir / "checkpoints" / "best.pt"
    last_ckpt = run_dir / "checkpoints" / "last.pt"

    if resume_from is not None:
        resumed_epoch, global_step, best_eval_loss = load_checkpoint(
            resume_from,
            image_encoder=image_encoder,
            text_projection=text_projection,
            logit_scale=logit_scale,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        start_epoch = resumed_epoch + 1
        if is_main:
            console.print(
                f"\n[bold cyan]Resumed from {resume_from}[/bold cyan] "
                f"(epoch {resumed_epoch}, step {global_step})"
            )

    if is_main:
        eff_batch = (
            config.dataset.batch_size * (dist_state.world_size if dist_state else 1) * accum_steps
        )
        console.print(
            f"\n[bold]Training for {opt_cfg.epochs} epochs "
            f"({total_steps} steps, effective batch={eff_batch})...[/bold]\n"
        )

    all_params = (
        list(image_encoder.parameters())
        + list(text_projection.parameters())
        + list(logit_scale.parameters())
    )

    def _save(path: Path, *, epoch: int, best: float) -> None:
        save_checkpoint(
            path,
            image_encoder=image_encoder,
            text_projection=text_projection,
            logit_scale=logit_scale,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            best_eval_loss=best,
            config=config,
        )

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=not is_main,
    )

    with progress:
        epoch_task = progress.add_task("Epochs", total=opt_cfg.epochs - start_epoch + 1)

        for epoch in range(start_epoch, opt_cfg.epochs + 1):
            if global_step >= total_steps:
                break
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            image_encoder.train()
            text_projection.train()
            epoch_losses: list[float] = []
            step_loss_accum = 0.0
            epoch_start = time.time()
            last_image_emb: torch.Tensor | None = None
            last_text_emb: torch.Tensor | None = None

            batch_task = progress.add_task(f"  Epoch {epoch}", total=len(train_loader))
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(train_loader):
                if global_step >= total_steps:
                    break

                is_accumulating = (batch_idx + 1) % accum_steps != 0 and (batch_idx + 1) != len(
                    train_loader
                )

                sync_ctx = ExitStack()
                if is_accumulating and use_ddp:
                    sync_ctx.enter_context(image_encoder.no_sync())
                    sync_ctx.enter_context(text_projection.no_sync())
                    sync_ctx.enter_context(logit_scale.no_sync())

                with sync_ctx:
                    all_image, all_text, image_emb, text_emb, all_broad = forward_step(
                        batch,
                        image_encoder,
                        text_projection,
                        text_cache,
                        device=device,
                        amp=config.runtime.amp,
                        use_cwa=opt_cfg.use_cwa,
                        use_ddp=use_ddp,
                        dist_cfg=dist_cfg,
                        dist_state=dist_state,
                    )
                    with autocast_context(device, config.runtime.amp):
                        loss = compute_loss(
                            opt_cfg.loss_type,
                            all_image,
                            all_text,
                            scale_param(logit_scale),
                            broad_samples=all_broad,
                        )
                        loss = loss / accum_steps
                    grad_scaler.scale(loss).backward()

                last_image_emb = image_emb.detach()
                last_text_emb = text_emb.detach()
                step_loss_accum += float(loss.detach().cpu().item()) * accum_steps

                if not is_accumulating:
                    grad_norm_before, grad_norm_after = optimizer_step(
                        optimizer,
                        scheduler,
                        grad_scaler,
                        all_params,
                        logit_scale,
                        grad_clip_norm=opt_cfg.grad_clip_norm,
                        logit_scale_max=opt_cfg.logit_scale_max,
                    )

                    global_step += 1
                    loss_val = step_loss_accum / accum_steps
                    step_loss_accum = 0.0
                    if use_ddp:
                        loss_val = all_reduce_scalar(loss_val)
                    epoch_losses.append(loss_val)

                    if is_main and global_step % config.runtime.log_every_steps == 0:
                        sp = scale_param(logit_scale)
                        current_lr = scheduler.get_last_lr()[0]
                        current_tau = sp.exp().item()
                        progress.console.print(
                            f"  [dim]step={global_step}/{total_steps} "
                            f"loss=[green]{loss_val:.5f}[/green] "
                            f"lr={current_lr:.6f} tau={current_tau:.4f}[/dim]"
                        )
                        with torch.no_grad():
                            logit_stats = compute_logit_stats(
                                sp.exp() * image_emb.detach() @ text_emb.detach().t()
                            )
                        logger.log_step(
                            global_step,
                            loss=loss_val,
                            lr=current_lr,
                            tau=current_tau,
                            grad_norm_before=grad_norm_before,
                            grad_norm_after=grad_norm_after,
                            logit_stats=logit_stats,
                            image_emb=image_emb.detach(),
                            text_emb=text_emb.detach(),
                        )
                        logger.log_model_health(
                            global_step,
                            image_encoder=image_encoder,
                            text_projection=text_projection,
                        )

                progress.update(batch_task, advance=1)

            progress.remove_task(batch_task)

            # --- Epoch summary ---
            train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
            train_metrics: dict[str, float | int] = {
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": train_loss,
                "epoch_seconds": float(time.time() - epoch_start),
            }

            # --- Evaluation ---
            eval_metrics: dict[str, float] | None = None
            if epoch % config.runtime.eval_every_epochs == 0 and is_main:
                eval_metrics = evaluate_epoch(
                    image_encoder,
                    text_projection,
                    text_cache,
                    val_loader,
                    device=device,
                    logit_scale=scale_param(logit_scale),
                    loss_type=opt_cfg.loss_type,
                    use_cwa=opt_cfg.use_cwa,
                    amp=config.runtime.amp,
                )
                train_metrics.update(eval_metrics)
                progress.console.print(
                    f"  Epoch {epoch}: train_loss=[green]{train_loss:.5f}[/green] "
                    f"eval_loss=[yellow]{eval_metrics['eval_loss']:.5f}[/yellow] "
                    f"i2t_R@1=[cyan]{eval_metrics.get('image_to_text_R@1', 0.0):.4f}[/cyan]"
                )

                if eval_metrics["eval_loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["eval_loss"]
                    _save(best_ckpt, epoch=epoch, best=best_eval_loss)
                    progress.console.print(
                        f"  [bold green]New best eval loss: {best_eval_loss:.5f} "
                        "(saved)[/bold green]"
                    )

            # --- Periodic save ---
            if is_main and (
                epoch % config.runtime.save_every_epochs == 0 or global_step >= total_steps
            ):
                _save(last_ckpt, epoch=epoch, best=best_eval_loss)

            if use_ddp:
                torch_dist.barrier()

            logger.log_epoch(epoch, train_metrics, eval_metrics)
            if (
                tb_cfg.histogram_every_epochs > 0
                and epoch % tb_cfg.histogram_every_epochs == 0
                and last_image_emb is not None
                and last_text_emb is not None
            ):
                logger.log_histograms(
                    epoch,
                    image_encoder=image_encoder,
                    text_projection=text_projection,
                    image_emb=last_image_emb,
                    text_emb=last_text_emb,
                )

            history.append(train_metrics)
            progress.update(epoch_task, advance=1)

    # --- Save history ---
    if is_main:
        pd.DataFrame(history).to_csv(run_dir / "metrics.csv", index=False)
    logger.close()

    if is_main:
        console.rule("[bold green]Training Complete")
        console.print(f"  Run dir: {run_dir}")
        console.print(f"  Best checkpoint: {best_ckpt if best_ckpt.exists() else 'N/A'}")

    return {
        "run_dir": run_dir,
        "train_wells": train_count,
        "val_wells": val_count,
        "best_checkpoint": best_ckpt if best_ckpt.exists() else None,
        "last_checkpoint": last_ckpt if last_ckpt.exists() else None,
        "metrics_path": run_dir / "metrics.csv",
    }
