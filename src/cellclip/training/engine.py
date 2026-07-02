"""Training loop for local CellCLIP.

Supports single-GPU and multi-GPU (DDP via ``torchrun``) training with optional
TensorBoard logging. Optimizer/scheduler/checkpointing live in
:mod:`cellclip.training.optim`; evaluation in :mod:`cellclip.training.evaluate`.
"""

from __future__ import annotations

import math
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
import torch.distributed as torch_dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cellclip.training.augment import FeatureBagAugmenter
from cellclip.training.config import CellCLIPTrainingConfig
from cellclip.training.dataset import prepare_datasets
from cellclip.training.evaluate import _move_optional_tokens, _move_tokens, evaluate_epoch
from cellclip.training.losses import compute_loss
from cellclip.training.model import CellCLIP, build_cellclip_model
from cellclip.training.optim import build_optimizer, build_scheduler, save_checkpoint
from morphoclip.training.config import TensorBoardConfig
from morphoclip.training.distributed import (
    DistributedState,
    all_reduce_scalar,
    cleanup_distributed,
    setup_distributed,
)
from morphoclip.training.metrics import compute_grad_norm, compute_logit_stats
from morphoclip.training.tb_logger import TrainingLogger
from morphoclip.utils.device import autocast_context, resolve_device, resolve_num_workers


def _build_augmenter(prepared, config: CellCLIPTrainingConfig) -> FeatureBagAugmenter | None:
    if config.dataset.within_well_interp_sites <= 0 and config.dataset.same_pert_interp_sites <= 0:
        return None
    return FeatureBagAugmenter(
        dataset=prepared.train_source_dataset,
        train_indices=list(prepared.train_dataset.indices),
        plate_contexts=prepared.plate_contexts,
        within_well_interp_sites=config.dataset.within_well_interp_sites,
        same_pert_interp_sites=config.dataset.same_pert_interp_sites,
        interp_alpha=config.dataset.interp_alpha,
    )


def train_cellclip(
    config: CellCLIPTrainingConfig,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Run local CellCLIP training and return a summary."""
    dist_cfg = config.distributed

    # --- Distributed setup ---
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
            config, run_dir=run_dir, device=device, dist_state=dist_state, is_main=is_main
        )
    finally:
        if dist_state is not None:
            cleanup_distributed()


def _train_loop(
    config: CellCLIPTrainingConfig,
    *,
    run_dir: Path,
    device: torch.device,
    dist_state: DistributedState | None,
    is_main: bool,
) -> dict[str, Any]:
    """Inner training loop."""
    dist_cfg = config.distributed
    use_ddp = dist_state is not None and dist_state.world_size > 1
    accum_steps = dist_cfg.gradient_accumulation_steps

    prepared = prepare_datasets(config.dataset, config.model)
    run_dir.mkdir(parents=True, exist_ok=True)
    if is_main:
        prepared.split_manifest.to_csv(run_dir / "split_manifest.csv", index=False)

    model: nn.Module = build_cellclip_model(config.model).to(device)
    augmenter = _build_augmenter(prepared, config)

    # --- DDP wrapping ---
    if use_ddp:
        assert dist_state is not None  # implied by use_ddp
        model = DDP(
            model,
            device_ids=[dist_state.local_rank],
            find_unused_parameters=dist_cfg.find_unused_parameters,
        )

    # --- Distributed sampler ---
    train_sampler: DistributedSampler | None = None
    if use_ddp:
        assert dist_state is not None  # implied by use_ddp
        train_sampler = DistributedSampler(
            prepared.train_dataset,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
        )
        # Rebuild train loader with sampler
        prepared.train_loader = DataLoader(
            prepared.train_dataset,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            sampler=train_sampler,
            collate_fn=prepared.train_loader.collate_fn,
            num_workers=resolve_num_workers(config.dataset.num_workers),
            pin_memory=config.dataset.pin_memory and device.type == "cuda",
        )

    optimizer = build_optimizer(model, config)

    steps_per_epoch = math.ceil(len(prepared.train_loader) / accum_steps)
    total_steps = max(1, steps_per_epoch * config.optimization.epochs)
    if config.runtime.max_train_steps is not None:
        total_steps = min(total_steps, config.runtime.max_train_steps)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=config.optimization.warmup_steps,
    )

    use_scaler = config.runtime.amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda" if use_scaler else "cpu", enabled=use_scaler)

    # --- TensorBoard logger ---
    rank = dist_state.rank if dist_state else 0
    logger = TrainingLogger(run_dir, cast(TensorBoardConfig, config.tensorboard), rank=rank)
    logger.log_config(config)

    history: list[dict[str, float | int]] = []
    best_eval_loss = float("inf")
    global_step = 0
    best_checkpoint_path = run_dir / "checkpoints" / "best.pt"
    last_checkpoint_path = run_dir / "checkpoints" / "last.pt"

    raw_model = cast(CellCLIP, model.module if hasattr(model, "module") else model)

    for epoch in range(1, config.optimization.epochs + 1):
        if global_step >= total_steps:
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses: list[float] = []
        epoch_start = time.time()

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(prepared.train_loader):
            if global_step >= total_steps:
                break

            is_accumulating = (batch_idx + 1) % accum_steps != 0 and (batch_idx + 1) != len(
                prepared.train_loader
            )

            features = batch["features"].to(device, non_blocking=True)
            site_mask = batch["site_mask"].to(device, non_blocking=True)
            text_tokens = _move_tokens(batch["text_tokens"], device)
            smiles_tokens = _move_optional_tokens(batch.get("smiles_tokens"), device)
            has_smiles = batch.get("has_smiles")
            if has_smiles is not None:
                has_smiles = has_smiles.to(device, non_blocking=True)
            if augmenter is not None:
                features = augmenter(
                    features,
                    site_mask=site_mask,
                    plates=batch["plates"],
                    wells=batch["wells"],
                    pert_info=batch["pert_info"],
                )

            # Skip gradient sync during accumulation
            sync_ctx = ExitStack()
            if is_accumulating and use_ddp:
                sync_ctx.enter_context(cast(DDP, model).no_sync())

            with sync_ctx:
                with autocast_context(device, config.runtime.amp):
                    pooled_images = raw_model.encode_mil(features)
                    outputs = model(
                        pooled_images,
                        text_tokens,
                        smiles=smiles_tokens,
                        has_smiles=has_smiles,
                    )
                    image_features, text_features, logit_scale = outputs[:3]
                    loss = compute_loss(
                        config.optimization.loss_type,
                        pooled_images,
                        image_features,
                        text_features,
                        logit_scale,
                    )
                    loss = loss / accum_steps

                grad_scaler.scale(loss).backward()

            if not is_accumulating:
                grad_scaler.unscale_(optimizer)
                all_params = list(model.parameters())
                grad_norm_before = float(
                    torch.nn.utils.clip_grad_norm_(all_params, config.optimization.grad_clip_norm)
                )
                grad_norm_after = compute_grad_norm(all_params)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                with torch.no_grad():
                    raw_model.logit_scale.data.clamp_(0, 4.6052)

                global_step += 1
                loss_value = float(loss.detach().cpu().item()) * accum_steps
                if use_ddp:
                    loss_value = all_reduce_scalar(loss_value)
                epoch_losses.append(loss_value)

                if is_main and global_step % config.runtime.log_every_steps == 0:
                    current_lr = float(scheduler.get_last_lr()[0])
                    current_tau = raw_model.logit_scale.exp().item()
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={loss_value:.5f} lr={current_lr:.6f} tau={current_tau:.4f}"
                    )

                    with torch.no_grad():
                        logit_stats = compute_logit_stats(
                            raw_model.logit_scale.exp()
                            * image_features.detach()
                            @ text_features.detach().t()
                        )
                    logger.log_step(
                        global_step,
                        loss=loss_value,
                        lr=current_lr,
                        tau=current_tau,
                        grad_norm_before=grad_norm_before,
                        grad_norm_after=grad_norm_after,
                        logit_stats=logit_stats,
                        image_emb=image_features.detach(),
                        text_emb=text_features.detach(),
                    )
                    logger.log_model_health(
                        global_step,
                        image_encoder=model,
                        text_projection=model,
                    )

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        train_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "epoch_seconds": float(time.time() - epoch_start),
        }

        # --- Evaluation (rank 0 only) ---
        eval_metrics: dict[str, float] | None = None
        if epoch % config.runtime.eval_every_epochs == 0 and is_main:
            eval_metrics = evaluate_epoch(
                model,
                prepared.eval_loader,
                device=device,
                loss_type=config.optimization.loss_type,
                amp=config.runtime.amp,
            )
            train_metrics.update(eval_metrics)
            print(
                f"epoch={epoch} eval_loss={eval_metrics['eval_loss']:.5f} "
                f"i2t_R@1={eval_metrics.get('image_to_text_R@1', 0.0):.4f} "
                f"t2i_R@1={eval_metrics.get('text_to_image_R@1', 0.0):.4f}"
            )

            if eval_metrics["eval_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_loss"]
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    best_eval_loss=best_eval_loss,
                    config=config,
                )

        if is_main and (
            epoch % config.runtime.save_every_epochs == 0 or global_step >= total_steps
        ):
            save_checkpoint(
                last_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_eval_loss=best_eval_loss,
                config=config,
            )

        if use_ddp:
            torch_dist.barrier()

        # --- TensorBoard epoch logging ---
        logger.log_epoch(epoch, train_metrics, eval_metrics)

        tb_cfg = config.tensorboard
        if (
            tb_cfg.enabled
            and tb_cfg.histogram_every_epochs > 0
            and epoch % tb_cfg.histogram_every_epochs == 0
        ):
            logger.log_histograms(
                epoch,
                image_encoder=model,
                text_projection=model,
                image_emb=image_features.detach() if "image_features" in dir() else None,
                text_emb=text_features.detach() if "text_features" in dir() else None,
            )

        history.append(train_metrics)

    if is_main:
        history_df = pd.DataFrame(history)
        history_df.to_csv(run_dir / "metrics.csv", index=False)
    logger.close()

    return {
        "run_dir": run_dir,
        "train_wells": len(prepared.train_dataset),
        "eval_wells": len(prepared.eval_dataset),
        "best_checkpoint": best_checkpoint_path if best_checkpoint_path.exists() else None,
        "last_checkpoint": last_checkpoint_path if last_checkpoint_path.exists() else None,
        "metrics_path": run_dir / "metrics.csv",
    }
