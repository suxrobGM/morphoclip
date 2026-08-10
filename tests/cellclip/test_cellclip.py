"""Tests for the local CellCLIP benchmark runtime."""

import torch

from cellclip.benchmark.checkpoint import load_cellclip_visual_encoder
from cellclip.model import CellCLIPVisualConfig, CellCLIPVisualEncoder


def test_visual_encoder_encode_image_shape() -> None:
    model = CellCLIPVisualEncoder(
        CellCLIPVisualConfig(
            embed_dim=512,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
        )
    )
    batch = torch.randn(3, 5, 16)
    output = model.encode_image(batch)
    assert output.shape == (3, 512)


def test_load_cellclip_visual_encoder_from_torch_checkpoint(tmp_path) -> None:
    reference = CellCLIPVisualEncoder(
        CellCLIPVisualConfig(
            embed_dim=512,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
        )
    )
    checkpoint_path = tmp_path / "cellclip_visual.pt"
    torch.save({"model": reference.state_dict()}, checkpoint_path)

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
    expected = reference.encode_image(sample)
    actual = loaded.encode_image(sample)
    assert torch.allclose(actual, expected)
