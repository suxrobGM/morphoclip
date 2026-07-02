"""Tests for CrossChannelFormer."""

import pytest
import torch

from morphoclip.models.cross_channel_former import CrossChannelFormer


class TestCrossChannelFormer:
    """Test suite for the CrossChannelFormer module."""

    def test_output_shape(self) -> None:
        model = CrossChannelFormer(embed_dim=64, num_layers=1, num_heads=4)
        x = torch.randn(2, 5, 64)
        out = model(x)
        assert out.shape == (2, 64)

    def test_default_config_output_shape(self) -> None:
        model = CrossChannelFormer()
        x = torch.randn(1, 5, 1024)
        out = model(x)
        assert out.shape == (1, 1024)

    def test_wrong_channel_count(self) -> None:
        model = CrossChannelFormer(embed_dim=64, num_layers=1, num_heads=4)
        x = torch.randn(2, 3, 64)
        with pytest.raises(ValueError, match="Expected 5 channels"):
            model(x)

    def test_wrong_embed_dim(self) -> None:
        model = CrossChannelFormer(embed_dim=64, num_layers=1, num_heads=4)
        x = torch.randn(2, 5, 128)
        with pytest.raises(ValueError, match="Expected embedding dim 64"):
            model(x)

    def test_deterministic_eval(self) -> None:
        model = CrossChannelFormer(embed_dim=64, num_layers=1, num_heads=4)
        model.eval()
        x = torch.randn(2, 5, 64)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_batch_size_one(self) -> None:
        model = CrossChannelFormer(embed_dim=64, num_layers=1, num_heads=4)
        x = torch.randn(1, 5, 64)
        out = model(x)
        assert out.shape == (1, 64)

    def test_custom_channels(self) -> None:
        model = CrossChannelFormer(
            embed_dim=64,
            num_layers=1,
            num_heads=4,
            input_channels=3,
        )
        x = torch.randn(2, 3, 64)
        out = model(x)
        assert out.shape == (2, 64)
