"""MorphoCLIP training orchestrator.

Wires together data loading, model creation, and the training loop. Model /
optimizer setup lives in :mod:`morphoclip.training.setup` and the per-epoch batch
loop in :mod:`morphoclip.training.loop`. Supports single-GPU and multi-GPU (DDP
via ``torchrun``) training.
"""

import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as torch_dist
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import (
    DistributedState,
    cleanup_distributed,
    setup_distributed,
)
from morphoclip.training.engine import (
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    scale_param,
)
from morphoclip.training.evaluate import evaluate_epoch
from morphoclip.training.loop import TrainContext, run_epoch
from morphoclip.training.setup import (
    build_and_wrap_models,
    build_optimization,
    log_config_summary,
    log_device_banner,
)
from morphoclip.training.tb_logger import TrainingLogger
from morphoclip.training.train_data import build_train_data
from morphoclip.utils.caching import load_cached_text_features

console = Console()


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
    world_size = dist_state.world_size if dist_state else 1

    if is_main:
        console.rule("[bold blue]MorphoCLIP Training")
        log_device_banner(console, device, use_ddp=use_ddp, world_size=world_size)

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
    image_encoder, text_projection, logit_scale = build_and_wrap_models(
        config, device, use_ddp=use_ddp, dist_state=dist_state
    )
    if is_main:
        log_config_summary(console, config, image_encoder, text_projection)

    # --- Optimizer & scheduler ---
    optimizer, scheduler, grad_scaler, total_steps = build_optimization(
        image_encoder,
        text_projection,
        logit_scale,
        config,
        device=device,
        num_batches=len(train_loader),
    )

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
        eff_batch = config.dataset.batch_size * world_size * accum_steps
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

    ctx = TrainContext(
        image_encoder=image_encoder,
        text_projection=text_projection,
        logit_scale=logit_scale,
        text_cache=text_cache,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
        all_params=all_params,
        logger=logger,
        config=config,
        device=device,
        dist_state=dist_state,
        use_ddp=use_ddp,
        accum_steps=accum_steps,
        is_main=is_main,
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
            epoch_start = time.time()

            batch_task = progress.add_task(f"  Epoch {epoch}", total=len(train_loader))
            result = run_epoch(
                ctx,
                train_loader,
                epoch=epoch,
                global_step=global_step,
                total_steps=total_steps,
                progress=progress,
                batch_task=batch_task,
            )
            global_step = result.global_step
            progress.remove_task(batch_task)

            # --- Epoch summary ---
            train_loss = float(sum(result.epoch_losses) / max(1, len(result.epoch_losses)))
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
                and result.last_image_emb is not None
                and result.last_text_emb is not None
            ):
                logger.log_histograms(
                    epoch,
                    image_encoder=image_encoder,
                    text_projection=text_projection,
                    image_emb=result.last_image_emb,
                    text_emb=result.last_text_emb,
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
