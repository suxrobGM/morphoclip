"""Unit tests for morphoclip.training.setup.build_optimization."""

import torch
from torch import nn

from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import LogitScaleModule
from morphoclip.training.setup import build_optimization


def _modules() -> tuple[nn.Module, nn.Module, nn.Module]:
    image_encoder = nn.Linear(4, 4)
    text_projection = nn.Linear(4, 4)
    logit_scale = LogitScaleModule(init_value=1.0, device=torch.device("cpu"))
    return image_encoder, text_projection, logit_scale


class TestBuildOptimization:
    def test_param_group_order_and_logit_scale_no_decay(self) -> None:
        image_encoder, text_projection, logit_scale = _modules()
        config = MorphoCLIPTrainingConfig()

        optimizer, _scheduler, _scaler, _total = build_optimization(
            image_encoder,
            text_projection,
            logit_scale,
            config,
            device=torch.device("cpu"),
            num_batches=10,
        )

        # split_params -> [decay, no_decay]; logit_scale group appended last.
        assert len(optimizer.param_groups) == 3
        logit_group = optimizer.param_groups[-1]
        assert logit_group["weight_decay"] == 0.0
        scale_param = logit_scale.scale
        assert any(p is scale_param for p in logit_group["params"])

    def test_total_steps_respects_accumulation_and_epochs(self) -> None:
        image_encoder, text_projection, logit_scale = _modules()
        config = MorphoCLIPTrainingConfig()
        config.optimization.epochs = 3
        config.distributed.gradient_accumulation_steps = 2
        config.runtime.max_train_steps = None

        _optimizer, _scheduler, _scaler, total_steps = build_optimization(
            image_encoder,
            text_projection,
            logit_scale,
            config,
            device=torch.device("cpu"),
            num_batches=10,
        )
        # ceil(10 / 2) * 3 epochs = 15
        assert total_steps == 15

    def test_max_train_steps_clamps_total(self) -> None:
        image_encoder, text_projection, logit_scale = _modules()
        config = MorphoCLIPTrainingConfig()
        config.optimization.epochs = 5
        config.distributed.gradient_accumulation_steps = 1
        config.runtime.max_train_steps = 4

        _optimizer, _scheduler, _scaler, total_steps = build_optimization(
            image_encoder,
            text_projection,
            logit_scale,
            config,
            device=torch.device("cpu"),
            num_batches=10,
        )
        assert total_steps == 4

    def test_scheduler_warmup_increases_lr(self) -> None:
        image_encoder, text_projection, logit_scale = _modules()
        config = MorphoCLIPTrainingConfig()
        config.optimization.epochs = 1
        config.optimization.warmup_steps = 5
        config.distributed.gradient_accumulation_steps = 1

        _optimizer, scheduler, _scaler, _total = build_optimization(
            image_encoder,
            text_projection,
            logit_scale,
            config,
            device=torch.device("cpu"),
            num_batches=10,
        )
        lr_start = scheduler.get_last_lr()[0]
        scheduler.step()
        scheduler.step()
        lr_after = scheduler.get_last_lr()[0]
        assert lr_after > lr_start
