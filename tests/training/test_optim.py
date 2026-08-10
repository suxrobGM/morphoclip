"""Tests for the optimizer and schedule both trainers now share.

MorphoCLIP and CellCLIP used to carry their own copy of the decay/no-decay split
and the warmup-cosine schedule. They differed in one way that matters and one
that did not:

* group ordering, which fixes the param indices in an optimizer state_dict, so
  a resumed run maps momentum onto the wrong tensors if it flips. That is why
  `split_params` takes `decay_first`, and both orderings are asserted here.
* an extra `logit_scale` clause in CellCLIP's predicate, which `ndim < 2`
  already covered. The test below proves it, so its removal is not a guess.
"""

import math

import pytest
import torch
from torch import nn

from cellclip.training.config import CellCLIPModelConfig, CellCLIPTrainingConfig
from cellclip.training.model import CellCLIP
from cellclip.training.optim import build_optimizer as cellclip_build_optimizer
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.optim import (
    build_optimizer,
    build_warmup_cosine_scheduler,
    split_params,
    unwrap,
    unwrap_state_dict,
)
from tests.support.stubs import FakeTextModel


def _morphoclip_modules() -> tuple[nn.Module, nn.Module]:
    from morphoclip.training.inference import build_models

    config = MorphoCLIPTrainingConfig()
    config.model.output_dim = 8
    config.model.ccf_layers = 1
    config.model.ccf_heads = 2
    return build_models(config, torch.device("cpu"))


def _cellclip_model(monkeypatch: pytest.MonkeyPatch) -> CellCLIP:
    monkeypatch.setattr(
        "cellclip.training.model.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: FakeTextModel(hidden_size=24),
    )
    return CellCLIP(
        CellCLIPModelConfig(
            embed_dim=32,
            vision_layers=1,
            vision_width=16,
            vision_heads=2,
            input_channels=5,
            text_model_name="fake",
            tokenizer_name="fake",
        )
    )


class TestSplitParams:
    def test_morphoclip_puts_the_decay_group_first(self) -> None:
        groups = split_params(*_morphoclip_modules(), weight_decay=0.2)
        assert [group["weight_decay"] for group in groups] == [0.2, 0.0]

    def test_cellclip_puts_the_no_decay_group_first(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = CellCLIPTrainingConfig()
        config.optimization.weight_decay = 0.2
        optimizer = cellclip_build_optimizer(_cellclip_model(monkeypatch), config)
        assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.2]

    def test_ordering_only_swaps_the_groups(self) -> None:
        modules = _morphoclip_modules()
        first, second = split_params(*modules, weight_decay=0.1)
        flipped = split_params(*modules, weight_decay=0.1, decay_first=False)
        assert [group["weight_decay"] for group in flipped] == [second["weight_decay"], 0.1]
        assert len(flipped[1]["params"]) == len(first["params"])

    def test_every_trainable_parameter_lands_in_exactly_one_group(self) -> None:
        modules = _morphoclip_modules()
        grouped = [p for group in split_params(*modules, weight_decay=0.1) for p in group["params"]]
        trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
        assert len(grouped) == len(trainable)
        assert {id(p) for p in grouped} == {id(p) for p in trainable}

    def test_frozen_parameters_are_left_out(self) -> None:
        linear = nn.Linear(4, 4)
        linear.weight.requires_grad_(False)
        decay, no_decay = split_params(linear, weight_decay=0.1)
        assert decay["params"] == []
        assert [p.shape for p in no_decay["params"]] == [linear.bias.shape]

    def test_one_dimensional_parameters_never_decay(self) -> None:
        """Covers the 0-d learnable temperature without naming it."""
        module = nn.Module()
        module.scale = nn.Parameter(torch.zeros(()))
        module.matrix = nn.Parameter(torch.zeros(3, 3))
        decay, no_decay = split_params(module, weight_decay=0.1)
        assert [p.ndim for p in decay["params"]] == [2]
        assert [p.ndim for p in no_decay["params"]] == [0]

    def test_the_cellclip_logit_scale_clause_was_redundant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = _cellclip_model(monkeypatch)
        no_decay = split_params(model, weight_decay=0.1, decay_first=False)[0]["params"]
        logit_scale = dict(model.named_parameters())["logit_scale"]
        assert any(p is logit_scale for p in no_decay)


class TestWarmupCosineScheduler:
    def _lr_lambda(self, *, total_steps: int, warmup_steps: int):
        optimizer = build_optimizer(
            split_params(*_morphoclip_modules(), weight_decay=0.0), MorphoCLIPTrainingConfig()
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer, total_steps=total_steps, warmup_steps=warmup_steps
        )
        return scheduler.lr_lambdas[0]

    def test_warmup_ramps_linearly_to_full_lr(self) -> None:
        lr_lambda = self._lr_lambda(total_steps=100, warmup_steps=4)
        assert [lr_lambda(step) for step in range(4)] == [0.25, 0.5, 0.75, 1.0]

    def test_cosine_decays_from_one_to_zero(self) -> None:
        lr_lambda = self._lr_lambda(total_steps=10, warmup_steps=0)
        assert lr_lambda(0) == pytest.approx(1.0)
        assert lr_lambda(5) == pytest.approx(0.5)
        assert lr_lambda(10) == pytest.approx(0.0, abs=1e-12)

    def test_the_schedule_never_goes_negative_past_the_end(self) -> None:
        lr_lambda = self._lr_lambda(total_steps=10, warmup_steps=2)
        assert lr_lambda(50) == pytest.approx(0.0, abs=1e-12)

    def test_a_zero_length_run_holds_the_base_lr(self) -> None:
        """Guards a division by zero on an empty train loader."""
        lr_lambda = self._lr_lambda(total_steps=0, warmup_steps=5)
        assert all(lr_lambda(step) == 1.0 for step in range(5))

    def test_warmup_longer_than_the_run_still_terminates(self) -> None:
        lr_lambda = self._lr_lambda(total_steps=3, warmup_steps=10)
        assert all(math.isfinite(lr_lambda(step)) for step in range(12))


class TestUnwrap:
    def test_a_plain_module_is_returned_as_is(self) -> None:
        module = nn.Linear(2, 2)
        assert unwrap(module) is module

    def test_a_ddp_style_wrapper_is_stripped(self) -> None:
        """Any module exposing `.module` is treated as wrapped, DDP included."""
        inner = nn.Linear(2, 2)
        wrapper = nn.Module()
        wrapper.module = inner
        assert unwrap(wrapper) is inner
        assert set(unwrap_state_dict(wrapper)) == set(inner.state_dict())
