"""Tests for WellFormer (joint site-channel transformer aggregation)."""

import pytest
import torch

from morphoclip.models.well_former import WellFormer


def _make_well_former() -> WellFormer:
    model = WellFormer(embed_dim=64, num_layers=1, num_heads=4, dropout=0.0)
    model.eval()
    return model


class TestWellFormer:
    def test_masking_ignores_padding(self) -> None:
        """Appending padded sites must not change the output."""
        model = _make_well_former()

        x_2 = torch.randn(1, 2, 5, 64)
        x_4 = torch.randn(1, 4, 5, 64)
        x_4[:, :2] = x_2

        torch.testing.assert_close(
            model(x_2, torch.tensor([[True, True]])),
            model(x_4, torch.tensor([[True, True, False, False]])),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_gradients_flow(self) -> None:
        model = _make_well_former()
        x = torch.randn(2, 3, 5, 64, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        model(x, mask).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    @pytest.mark.parametrize(
        ("channels", "dim", "mask_sites", "message"),
        [
            (3, 64, 2, "Expected 5 channels"),
            (5, 128, 2, "Expected embedding dim 64"),
            (5, 64, 3, "site_mask"),
        ],
    )
    def test_malformed_input_raises(
        self, channels: int, dim: int, mask_sites: int, message: str
    ) -> None:
        model = _make_well_former()
        with pytest.raises(ValueError, match=message):
            model(torch.randn(1, 2, channels, dim), torch.ones(1, mask_sites, dtype=torch.bool))
