"""Per-epoch batch loop for MorphoCLIP training.

:func:`run_epoch` owns the batch loop for one epoch; the coordinator in
:mod:`morphoclip.training.trainer` threads ``global_step`` in and out via
:class:`EpochResult` and reads back the last embeddings for histogram logging.
"""

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, cast

import torch
from rich.progress import Progress, TaskID
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from morphoclip.training.config import (
    GRAD_CLIP_NORM,
    LOGIT_SCALE_MAX,
    MorphoCLIPTrainingConfig,
)
from morphoclip.training.distributed import DistributedState, all_reduce_scalar
from morphoclip.training.engine import (
    autocast_context,
    forward_step,
    optimizer_step,
    scale_param,
)
from morphoclip.training.losses import compute_loss, replicate_image_loss
from morphoclip.training.metrics import compute_logit_stats
from morphoclip.training.tb_logger import TrainingLogger


@dataclass
class TrainContext:
    """Read-only handles the batch loop needs (avoids a 15-argument signature)."""

    image_encoder: nn.Module
    text_projection: nn.Module
    logit_scale: nn.Module
    text_cache: dict
    optimizer: AdamW
    scheduler: LambdaLR
    grad_scaler: Any
    all_params: list[nn.Parameter]
    logger: TrainingLogger
    config: MorphoCLIPTrainingConfig
    device: torch.device
    dist_state: DistributedState | None
    use_ddp: bool
    accum_steps: int
    is_main: bool


@dataclass
class EpochResult:
    """Mutable per-epoch outputs the coordinator reads back after run_epoch."""

    global_step: int
    epoch_losses: list[float]
    last_image_emb: torch.Tensor | None
    last_text_emb: torch.Tensor | None


def enter_no_sync(stack: ExitStack, *modules: nn.Module) -> None:
    """Enter ``no_sync()`` for each DDP-wrapped module (skips gradient sync)."""
    for module in modules:
        stack.enter_context(cast(DDP, module).no_sync())


def image_image_scale(
    logit_scale: nn.Module,
    *,
    img_img_temperature: float | None,
) -> torch.Tensor:
    """Resolve the linear scale for the replicate image-image term.

    Defaults to the shared learnable temperature; ``img_img_temperature`` pins
    it to ``1 / temperature`` instead.
    """
    if img_img_temperature is None:
        return scale_param(logit_scale).exp()
    scale = scale_param(logit_scale)
    return torch.tensor(1.0 / img_img_temperature, device=scale.device, dtype=scale.dtype)


def run_epoch(
    ctx: TrainContext,
    train_loader,
    *,
    epoch: int,
    global_step: int,
    total_steps: int,
    progress: Progress,
    batch_task: TaskID,
) -> EpochResult:
    """Run one training epoch's batch loop, returning updated step + last embeddings."""
    config = ctx.config
    opt_cfg = config.optimization
    image_encoder = ctx.image_encoder
    text_projection = ctx.text_projection
    logit_scale = ctx.logit_scale
    device = ctx.device
    use_ddp = ctx.use_ddp
    accum_steps = ctx.accum_steps
    grad_scaler = ctx.grad_scaler
    logger = ctx.logger

    epoch_losses: list[float] = []
    step_loss_accum = 0.0
    step_img_loss_accum = 0.0
    last_image_emb: torch.Tensor | None = None
    last_text_emb: torch.Tensor | None = None

    ctx.optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(train_loader):
        if global_step >= total_steps:
            break

        is_accumulating = (batch_idx + 1) % accum_steps != 0 and (batch_idx + 1) != len(
            train_loader
        )

        img_img_value = 0.0
        sync_ctx = ExitStack()
        if is_accumulating and use_ddp:
            enter_no_sync(sync_ctx, image_encoder, text_projection, logit_scale)

        with sync_ctx:
            all_image, all_text, image_emb, text_emb, all_broad, all_targets = forward_step(
                batch,
                image_encoder,
                text_projection,
                ctx.text_cache,
                device=device,
                amp=config.runtime.amp,
                use_cwa=opt_cfg.use_cwa,
                use_ddp=use_ddp,
                dist_state=ctx.dist_state,
            )
            with autocast_context(device, config.runtime.amp):
                loss = compute_loss(
                    opt_cfg.loss_type,
                    all_image,
                    all_text,
                    scale_param(logit_scale),
                    broad_samples=all_broad,
                    target_keys=all_targets,
                    target_weight=opt_cfg.target_weight,
                )
                # Added here rather than inside compute_loss so eval_loss stays
                # text alignment only.
                if opt_cfg.lambda_img > 0.0:
                    img_img_loss = replicate_image_loss(
                        all_image,
                        image_image_scale(
                            logit_scale,
                            img_img_temperature=opt_cfg.img_img_temperature,
                        ),
                        broad_samples=all_broad,
                    )
                    img_img_value = float(img_img_loss.detach())
                    loss = loss + opt_cfg.lambda_img * img_img_loss
                loss = loss / accum_steps
            grad_scaler.scale(loss).backward()

        last_image_emb = image_emb.detach()
        last_text_emb = text_emb.detach()
        step_loss_accum += float(loss.detach().cpu().item()) * accum_steps
        step_img_loss_accum += img_img_value

        if not is_accumulating:
            grad_norm_before, grad_norm_after = optimizer_step(
                ctx.optimizer,
                ctx.scheduler,
                grad_scaler,
                ctx.all_params,
                logit_scale,
                grad_clip_norm=GRAD_CLIP_NORM,
                logit_scale_max=LOGIT_SCALE_MAX,
            )

            global_step += 1
            loss_val = step_loss_accum / accum_steps
            img_img_val = step_img_loss_accum / accum_steps
            step_loss_accum = 0.0
            step_img_loss_accum = 0.0
            if use_ddp:
                loss_val = all_reduce_scalar(loss_val)
            epoch_losses.append(loss_val)

            if ctx.is_main and global_step % config.runtime.log_every_steps == 0:
                sp = scale_param(logit_scale)
                current_lr = float(ctx.scheduler.get_last_lr()[0])
                current_tau = sp.exp().item()
                img_img_note = (
                    f" img_img=[magenta]{img_img_val:.5f}[/magenta]"
                    if opt_cfg.lambda_img > 0.0
                    else ""
                )
                progress.console.print(
                    f"  [dim]step={global_step}/{total_steps} "
                    f"loss=[green]{loss_val:.5f}[/green]{img_img_note} "
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
                    img_img_loss=img_img_val if opt_cfg.lambda_img > 0.0 else None,
                )
                logger.log_model_health(
                    global_step,
                    image_encoder=image_encoder,
                    text_projection=text_projection,
                )

        progress.update(batch_task, advance=1)

    return EpochResult(
        global_step=global_step,
        epoch_losses=epoch_losses,
        last_image_emb=last_image_emb,
        last_text_emb=last_text_emb,
    )
