"""Tests for WellFormer (joint site-channel transformer aggregation)."""

import pytest
import torch

from morphoclip.models.well_former import WellFormer


def _make_well_former() -> WellFormer:
    model = WellFormer(embed_dim=64, num_layers=1, num_heads=4, dropout=0.0)
    model.eval()
    return model


class TestWellFormer:
    """Test suite for the WellFormer module."""

    def test_output_shape(self) -> None:
        model = _make_well_former()
        x = torch.randn(2, 4, 5, 64)
        mask = torch.ones(2, 4, dtype=torch.bool)
        assert model(x, mask).shape == (2, 64)

    def test_masking_ignores_padding(self) -> None:
        """Appending padded sites must not change the output."""
        model = _make_well_former()

        x_2 = torch.randn(1, 2, 5, 64)
        mask_2 = torch.tensor([[True, True]])

        x_4 = torch.zeros(1, 4, 5, 64)
        x_4[:, :2] = x_2
        x_4[:, 2:] = torch.randn(1, 2, 5, 64)  # garbage in the padded slots
        mask_4 = torch.tensor([[True, True, False, False]])

        torch.testing.assert_close(model(x_2, mask_2), model(x_4, mask_4), atol=1e-5, rtol=1e-5)

    def test_site_permutation_invariance(self) -> None:
        """Sites are unordered: shuffling them must not change the output."""
        model = _make_well_former()
        x = torch.randn(1, 4, 5, 64)
        mask = torch.ones(1, 4, dtype=torch.bool)

        perm = torch.tensor([2, 0, 3, 1])
        out = model(x, mask)
        out_perm = model(x[:, perm], mask)
        torch.testing.assert_close(out, out_perm, atol=1e-5, rtol=1e-5)

    def test_channel_order_matters(self) -> None:
        """Channel-type embeddings make channels distinguishable."""
        model = _make_well_former()
        x = torch.randn(1, 3, 5, 64)
        mask = torch.ones(1, 3, dtype=torch.bool)

        perm = torch.tensor([4, 3, 2, 1, 0])
        out = model(x, mask)
        out_perm = model(x[:, :, perm], mask)
        assert not torch.allclose(out, out_perm, atol=1e-4)

    def test_single_site(self) -> None:
        model = _make_well_former()
        x = torch.randn(1, 1, 5, 64)
        mask = torch.tensor([[True]])
        assert model(x, mask).shape == (1, 64)

    def test_gradients_flow(self) -> None:
        model = WellFormer(embed_dim=64, num_layers=1, num_heads=4, dropout=0.0)
        x = torch.randn(2, 3, 5, 64, requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        model(x, mask).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    def test_wrong_channel_count_raises(self) -> None:
        model = _make_well_former()
        x = torch.randn(1, 2, 3, 64)
        mask = torch.ones(1, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="Expected 5 channels"):
            model(x, mask)

    def test_wrong_embed_dim_raises(self) -> None:
        model = _make_well_former()
        x = torch.randn(1, 2, 5, 128)
        mask = torch.ones(1, 2, dtype=torch.bool)
        with pytest.raises(ValueError, match="Expected embedding dim 64"):
            model(x, mask)

    def test_mask_shape_mismatch_raises(self) -> None:
        model = _make_well_former()
        x = torch.randn(1, 2, 5, 64)
        mask = torch.ones(1, 3, dtype=torch.bool)
        with pytest.raises(ValueError, match="site_mask"):
            model(x, mask)
