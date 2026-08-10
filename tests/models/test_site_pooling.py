"""Tests for AttentionSitePooling (gated-attention MIL over sites)."""

import pytest
import torch

from morphoclip.models.site_pooling import AttentionSitePooling


def _make_pooling() -> AttentionSitePooling:
    pooling = AttentionSitePooling(embed_dim=64, hidden_dim=16, dropout=0.0)
    pooling.eval()
    return pooling


class TestAttentionSitePooling:
    def test_weights_sum_to_one_over_real_sites(self) -> None:
        pooling = _make_pooling()
        x = torch.randn(2, 4, 64)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        weights = pooling.compute_weights(x, mask)
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0)
        assert torch.all(weights[0, 2:] == 0.0)

    def test_masking_ignores_padding(self) -> None:
        """Appending padded sites must not change the pooled output."""
        pooling = _make_pooling()

        x_2 = torch.randn(1, 2, 64)
        x_4 = torch.randn(1, 4, 64)
        x_4[:, :2] = x_2

        torch.testing.assert_close(
            pooling(x_2, torch.tensor([[True, True]])),
            pooling(x_4, torch.tensor([[True, True, False, False]])),
            atol=1e-6,
            rtol=1e-6,
        )

    def test_single_site_returns_that_site(self) -> None:
        pooling = _make_pooling()
        x = torch.randn(1, 1, 64)
        torch.testing.assert_close(
            pooling(x, torch.tensor([[True]])), x[:, 0], atol=1e-6, rtol=1e-6
        )

    def test_all_padded_row_pools_to_zero_without_nans(self) -> None:
        pooling = _make_pooling()
        x = torch.randn(2, 3, 64)
        mask = torch.tensor([[True, True, False], [False, False, False]])
        out = pooling(x, mask)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out[1], torch.zeros(64))

    def test_gradients_flow(self) -> None:
        pooling = _make_pooling()
        x = torch.randn(2, 4, 64, requires_grad=True)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        pooling(x, mask).sum().backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        for name, param in pooling.named_parameters():
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name

    def test_dim_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected embedding dim 64"):
            _make_pooling()(torch.randn(1, 2, 32), torch.ones(1, 2, dtype=torch.bool))
