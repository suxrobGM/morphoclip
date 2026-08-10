"""Tests for morphoclip.training.setup.build_optimization."""

import pytest
import torch
from torch import nn

from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import LogitScaleModule
from morphoclip.training.setup import build_optimization

CPU = torch.device("cpu")


@pytest.fixture
def modules() -> tuple[nn.Module, nn.Module, LogitScaleModule]:
    return nn.Linear(4, 4), nn.Linear(4, 4), LogitScaleModule(init_value=1.0, device=CPU)


def _build(
    modules: tuple[nn.Module, nn.Module, LogitScaleModule],
    config: MorphoCLIPTrainingConfig,
    *,
    num_batches: int = 10,
):
    return build_optimization(*modules, config, device=CPU, num_batches=num_batches)


class TestBuildOptimization:
    def test_the_logit_scale_gets_its_own_no_decay_group_last(
        self, modules: tuple[nn.Module, nn.Module, LogitScaleModule]
    ) -> None:
        """split_params gives [decay, no_decay]; the logit_scale group is appended."""
        optimizer, _scheduler, _scaler, _total = _build(modules, MorphoCLIPTrainingConfig())

        assert len(optimizer.param_groups) == 3
        logit_group = optimizer.param_groups[-1]
        assert logit_group["weight_decay"] == 0.0
        assert any(p is modules[2].scale for p in logit_group["params"])

    @pytest.mark.parametrize(
        ("epochs", "accumulation", "max_train_steps", "expected"),
        [(3, 2, None, 15), (5, 1, 4, 4)],
    )
    def test_total_steps_counts_accumulated_batches_and_honours_the_cap(
        self,
        modules: tuple[nn.Module, nn.Module, LogitScaleModule],
        epochs: int,
        accumulation: int,
        max_train_steps: int | None,
        expected: int,
    ) -> None:
        config = MorphoCLIPTrainingConfig()
        config.optimization.epochs = epochs
        config.distributed.gradient_accumulation_steps = accumulation
        config.runtime.max_train_steps = max_train_steps

        _optimizer, _scheduler, _scaler, total_steps = _build(modules, config)
        assert total_steps == expected

    def test_the_scheduler_is_wired_with_warmup(
        self, modules: tuple[nn.Module, nn.Module, LogitScaleModule]
    ) -> None:
        config = MorphoCLIPTrainingConfig()
        config.optimization.epochs = 1
        config.optimization.warmup_steps = 5
        config.distributed.gradient_accumulation_steps = 1

        optimizer, scheduler, _scaler, _total = _build(modules, config)
        lr_start = scheduler.get_last_lr()[0]
        for _ in range(2):
            optimizer.step()
            scheduler.step()
        assert scheduler.get_last_lr()[0] > lr_start
