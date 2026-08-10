"""Pin the two optimizer/scheduler implementations before they are merged.

MorphoCLIP and CellCLIP each carry their own copy of the decay/no-decay split
and the warmup-cosine schedule. They differ in two ways that are easy to erase
by accident and impossible to notice afterwards:

* group ordering, which decides the param indices in an optimizer state_dict, and
* an extra ``logit_scale`` clause in CellCLIP's decay predicate.

These tests record both, so merging the implementations has to be a deliberate
choice rather than a side effect.
"""

import pytest
import torch
from torch import nn

from cellclip.training.config import CellCLIPModelConfig, CellCLIPTrainingConfig
from cellclip.training.model import CellCLIP
from cellclip.training.optim import build_optimizer as cellclip_build_optimizer
from cellclip.training.optim import build_scheduler as cellclip_build_scheduler
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.engine import build_optimizer, build_scheduler, split_params
from morphoclip.training.inference import build_models
from tests.cellclip.conftest import FakeTextModel


def _morphoclip_modules() -> tuple[nn.Module, nn.Module]:
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


def test_morphoclip_puts_the_decay_group_first() -> None:
    image_encoder, text_projection = _morphoclip_modules()
    groups = split_params(image_encoder, text_projection, weight_decay=0.2)
    assert [group["weight_decay"] for group in groups] == [0.2, 0.0]


def test_cellclip_puts_the_no_decay_group_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opposite order to MorphoCLIP. A merge must keep it, or resume maps momentum wrong."""
    config = CellCLIPTrainingConfig()
    config.optimization.weight_decay = 0.2
    optimizer = cellclip_build_optimizer(_cellclip_model(monkeypatch), config)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.2]


def test_cellclip_logit_scale_clause_is_redundant(monkeypatch: pytest.MonkeyPatch) -> None:
    """`logit_scale` is 0-d, so `ndim < 2` already excludes it from weight decay.

    Recorded so the extra clause can be dropped in the merge without guessing.
    """
    model = _cellclip_model(monkeypatch)
    logit_scale = dict(model.named_parameters())["logit_scale"]
    assert logit_scale.ndim < 2

    def no_decay_names(*, with_logit_scale_clause: bool) -> set[str]:
        names = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lowered = name.lower()
            excluded = param.ndim < 2 or "bias" in lowered or "ln" in lowered or "norm" in lowered
            if with_logit_scale_clause:
                excluded = excluded or "logit_scale" in lowered
            if excluded:
                names.add(name)
        return names

    assert no_decay_names(with_logit_scale_clause=True) == no_decay_names(
        with_logit_scale_clause=False
    )


@pytest.mark.parametrize(("total_steps", "warmup_steps"), [(10, 3), (100, 0), (7, 7), (50, 1)])
def test_the_two_schedulers_agree_exactly(total_steps: int, warmup_steps: int) -> None:
    """Bit-identical, not approximately equal, so the merge is provably a no-op.

    Compares the lr_lambda functions rather than driving two optimizers, since
    the lambda is the whole of what differs between the two implementations.
    """
    image_encoder, text_projection = _morphoclip_modules()
    groups = split_params(image_encoder, text_projection, weight_decay=0.0)
    config = MorphoCLIPTrainingConfig()

    morpho = build_scheduler(
        build_optimizer(groups, config), total_steps=total_steps, warmup_steps=warmup_steps
    ).lr_lambdas[0]
    cell = cellclip_build_scheduler(
        build_optimizer(groups, config), total_steps=total_steps, warmup_steps=warmup_steps
    ).lr_lambdas[0]

    for step in range(total_steps + 2):
        assert morpho(step) == cell(step), f"diverged at step {step}"
