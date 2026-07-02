"""TensorBoard logging for MorphoCLIP training.

Wraps ``SummaryWriter`` for training scalars, embedding diagnostics,
weight histograms, and similarity heatmaps.  Silent no-op when
``rank != 0`` or TensorBoard is disabled in config.
"""

from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from morphoclip.training.config import TensorBoardConfig
from morphoclip.training.metrics import (
    compute_alignment,
    compute_intra_batch_similarity,
    compute_param_norm,
    compute_uniformity,
)

_CUSTOM_SCALARS_LAYOUT = {
    "Overfitting": {
        "Loss (train vs eval)": [
            "Multiline",
            ["epoch/train_loss", "epoch/eval_loss", "epoch/loss_gap"],
        ],
    },
    "Retrieval": {
        "R@1": [
            "Multiline",
            [
                "retrieval/image_to_text_R@1",
                "retrieval/text_to_image_R@1",
                "retrieval/image_to_text_R@5",
            ],
        ],
    },
    "Embedding Health": {
        "Alignment & Uniformity": [
            "Multiline",
            ["embedding/alignment", "embedding/uniformity", "embedding/intra_batch_similarity"],
        ],
    },
    "Gradients": {
        "Gradient norms": [
            "Multiline",
            ["train/grad_norm_before_clip", "train/grad_norm_after_clip", "train/loss"],
        ],
    },
}


class TrainingLogger:
    """Unified training logger with TensorBoard backend.

    Silent no-op when ``rank != 0``.
    """

    def __init__(
        self,
        run_dir: Path,
        tb_config: TensorBoardConfig,
        *,
        rank: int = 0,
    ) -> None:
        self._enabled = rank == 0
        self._flush_every = tb_config.flush_every_steps
        self._histogram_every = tb_config.histogram_every_epochs
        self._writer: SummaryWriter | None = None

        if self._enabled:
            log_dir = run_dir / "tensorboard"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

    def log_config(self, config: Any) -> None:
        """Write training config and custom scalars layout."""
        w = self._writer
        if w is None:
            return
        config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
        w.add_text(
            "config",
            f"```yaml\n{yaml.dump(config_dict, default_flow_style=False)}```",
            global_step=0,
        )
        try:
            w.add_custom_scalars(_CUSTOM_SCALARS_LAYOUT)
        except Exception:
            pass

    def log_step(
        self,
        step: int,
        *,
        loss: float,
        lr: float,
        tau: float,
        grad_norm_before: float,
        grad_norm_after: float,
        logit_stats: dict[str, float],
        image_emb: torch.Tensor | None = None,
        text_emb: torch.Tensor | None = None,
    ) -> None:
        """Log per-step training scalars and embedding diagnostics."""
        w = self._writer
        if w is None:
            return

        w.add_scalar("train/loss", loss, step)
        w.add_scalar("train/lr", lr, step)
        w.add_scalar("train/tau", tau, step)
        w.add_scalar("train/grad_norm_before_clip", grad_norm_before, step)
        w.add_scalar("train/grad_norm_after_clip", grad_norm_after, step)
        w.add_scalar("train/logit_mean", logit_stats["mean"], step)
        w.add_scalar("train/logit_std", logit_stats["std"], step)
        w.add_scalar("train/logit_max", logit_stats["max"], step)

        if image_emb is not None and text_emb is not None:
            img = image_emb.detach().float().cpu()
            txt = text_emb.detach().float().cpu()
            w.add_scalar("embedding/alignment", compute_alignment(img, txt), step)
            w.add_scalar("embedding/uniformity", compute_uniformity(img), step)
            w.add_scalar(
                "embedding/intra_batch_similarity", compute_intra_batch_similarity(img), step
            )

        if step % self._flush_every == 0:
            w.flush()

    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float | int],
        eval_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log per-epoch train/eval scalars and retrieval metrics."""
        w = self._writer
        if w is None:
            return

        train_loss = train_metrics.get("train_loss")
        if train_loss is not None:
            w.add_scalar("epoch/train_loss", train_loss, epoch)

        if eval_metrics is not None:
            eval_loss = eval_metrics.get("eval_loss")
            if eval_loss is not None:
                w.add_scalar("epoch/eval_loss", eval_loss, epoch)
                if train_loss is not None:
                    w.add_scalar("epoch/loss_gap", eval_loss - train_loss, epoch)

            for key, value in eval_metrics.items():
                if key == "eval_loss":
                    continue
                w.add_scalar(f"retrieval/{key}", value, epoch)

        w.flush()

    def log_model_health(
        self,
        step: int,
        *,
        image_encoder: nn.Module,
        text_projection: nn.Module,
    ) -> None:
        """Log parameter norms for model health monitoring."""
        w = self._writer
        if w is None:
            return
        w.add_scalar("health/image_encoder_param_norm", compute_param_norm(image_encoder), step)
        w.add_scalar("health/text_projection_param_norm", compute_param_norm(text_projection), step)

    def log_histograms(
        self,
        epoch: int,
        *,
        image_encoder: nn.Module,
        text_projection: nn.Module,
        image_emb: torch.Tensor | None = None,
        text_emb: torch.Tensor | None = None,
    ) -> None:
        """Log weight and embedding histograms (called every histogram_every_epochs)."""
        w = self._writer
        if w is None:
            return

        for name, param in image_encoder.named_parameters():
            if param.requires_grad:
                w.add_histogram(f"weights/image_encoder/{name}", param.data, epoch)
        for name, param in text_projection.named_parameters():
            if param.requires_grad:
                w.add_histogram(f"weights/text_projection/{name}", param.data, epoch)

        if image_emb is not None:
            w.add_histogram("embeddings/image", image_emb.detach().float().cpu(), epoch)
        if text_emb is not None:
            w.add_histogram("embeddings/text", text_emb.detach().float().cpu(), epoch)

        if image_emb is not None and text_emb is not None:
            img = image_emb.detach().float().cpu()
            txt = text_emb.detach().float().cpu()
            sim = img @ txt.t()
            sim_min, sim_max = sim.min(), sim.max()
            sim_normalized = (
                (sim - sim_min) / (sim_max - sim_min)
                if sim_max > sim_min
                else torch.zeros_like(sim)
            )
            w.add_image("similarity_matrix", sim_normalized.unsqueeze(0), epoch)

        w.flush()

    def close(self) -> None:
        """Flush and close the TensorBoard writer."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
