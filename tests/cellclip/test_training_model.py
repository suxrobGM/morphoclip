"""Tests for the CellCLIP model (checkpoint compatibility, trainable text encoder)."""

from pathlib import Path

import pytest
import torch

from cellclip.benchmark.checkpoint import load_cellclip_visual_encoder
from cellclip.training.config import CellCLIPModelConfig
from cellclip.training.model import CellCLIP
from tests.cellclip.conftest import FakeTextModel


def test_training_checkpoint_is_visual_loader_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "cellclip.training.model.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: FakeTextModel(hidden_size=24),
    )

    model = CellCLIP(
        CellCLIPModelConfig(
            embed_dim=512,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake",
            tokenizer_name="fake",
        )
    )
    checkpoint_path = tmp_path / "train_ckpt.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    loaded = load_cellclip_visual_encoder(
        model_path=str(checkpoint_path),
        device="cpu",
        input_dim=16,
        embed_dim=512,
        vision_layers=2,
        vision_heads=4,
        input_channels=5,
    )

    sample = torch.randn(2, 5, 16)
    expected = model.encode_image(sample)
    actual = loaded.encode_image(sample)
    assert torch.allclose(actual, expected)

    bag = torch.randn(2, 3, 5, 16)
    expected_bag = model.encode_image(model.encode_mil(bag))
    actual_bag = loaded.encode_image(loaded.encode_mil(bag))
    assert torch.allclose(actual_bag, expected_bag)


def test_cellclip_text_encoder_is_trainable_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "cellclip.training.model.AutoModel.from_pretrained",
        lambda *_args, **_kwargs: FakeTextModel(hidden_size=24),
    )

    model = CellCLIP(
        CellCLIPModelConfig(
            embed_dim=64,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            text_model_name="fake",
            tokenizer_name="fake",
        )
    )

    assert all(param.requires_grad for param in model.text.parameters())

    model.train()
    assert model.training is True
    assert model.text.training is True

    text_tokens = {
        "input_ids": torch.randint(0, 100, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    text_features = model.encode_text(text_tokens)
    text_features.sum().backward()

    assert model.text_proj.weight.grad is not None
    assert any(param.grad is not None for param in model.text.parameters())
