"""Tests for local CellCLIP benchmark export helpers."""

from __future__ import annotations

import numpy as np
import torch

from cellclip.benchmark.export import encode_well
from cellclip.benchmark.model import CellCLIPVisualConfig, CellCLIPVisualEncoder


def test_encode_well_matches_trainer_pool_then_encode_path() -> None:
    model = CellCLIPVisualEncoder(
        CellCLIPVisualConfig(
            embed_dim=8,
            vision_layers=2,
            vision_width=16,
            vision_heads=4,
            input_channels=5,
            pooling="attention",
        )
    )
    model.eval()

    sites = torch.randn(4, 5, 16)
    with torch.no_grad():
        expected = model.encode_image(model.encode_mil(sites.unsqueeze(0))).squeeze(0).numpy()

    actual = encode_well(model, sites, device="cpu", site_batch_size=1)

    assert isinstance(actual, np.ndarray)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected)
