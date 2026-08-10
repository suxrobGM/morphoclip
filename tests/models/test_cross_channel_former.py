"""Tests for CrossChannelFormer."""

import pytest
import torch

from morphoclip.models.cross_channel_former import CrossChannelFormer


class TestCrossChannelFormer:
    @pytest.mark.parametrize("input_channels", [5, 3])
    def test_channel_tokens_collapse_to_one_row_per_site(self, input_channels: int) -> None:
        model = CrossChannelFormer(
            embed_dim=64, num_layers=1, num_heads=4, input_channels=input_channels
        )
        assert model(torch.randn(2, input_channels, 64)).shape == (2, 64)

    @pytest.mark.parametrize(
        ("channels", "dim", "message"),
        [(3, 64, "Expected 5 channels"), (5, 128, "Expected embedding dim 64")],
    )
    def test_malformed_input_raises(self, channels: int, dim: int, message: str) -> None:
        model = CrossChannelFormer(embed_dim=64, num_layers=1, num_heads=4)
        with pytest.raises(ValueError, match=message):
            model(torch.randn(2, channels, dim))
