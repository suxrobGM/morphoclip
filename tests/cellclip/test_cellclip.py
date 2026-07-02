"""Tests for the local CellCLIP benchmark runtime."""

import pandas as pd
import torch

from cellclip.benchmark.checkpoint import load_cellclip_visual_encoder
from cellclip.benchmark.export import encode_well, negcon_center_profiles
from cellclip.benchmark.model import CellCLIPVisualConfig, CellCLIPVisualEncoder


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


def test_encode_well_mean_pools_sites() -> None:
    class DummyEncoder:
        def encode_mil(self, batch: torch.Tensor) -> torch.Tensor:
            # MIL pooling: [B, sites, channels, D] -> [B, channels, D]
            return batch.mean(dim=1)

        def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
            # Image encoding: [B, channels, D] -> [B, D]
            return batch.mean(dim=1)

    sites = torch.randn(3, 5, 4)
    embedding = encode_well(DummyEncoder(), sites, device="cpu", site_batch_size=2)
    # encode_mil([1, 3, 5, 4]) -> [1, 5, 4] -> encode_image([1, 5, 4]) -> [1, 4] -> squeeze -> (4,)
    assert embedding.shape == (4,)


def test_negcon_center_profiles_uses_negative_controls() -> None:
    profiles = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02", "A03"],
            "Metadata_control_type": ["negcon", "negcon", "trt"],
            "feature_0000": [1.0, 3.0, 8.0],
            "feature_0001": [2.0, 4.0, 10.0],
        }
    )

    centered = negcon_center_profiles(profiles)

    negcon = centered.loc[
        centered["Metadata_control_type"] == "negcon", ["feature_0000", "feature_0001"]
    ]
    assert torch.allclose(
        torch.tensor(negcon.to_numpy(dtype="float32").mean(axis=0)),
        torch.zeros(2),
    )
    assert centered.loc[2, "feature_0000"] == 6.0
    assert centered.loc[2, "feature_0001"] == 7.0
