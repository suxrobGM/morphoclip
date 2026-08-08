"""Tests for MorphoCLIPImageEncoder."""

import pytest
import torch
import torch.nn.functional as F

from morphoclip.models.image_encoder import AGGREGATORS, MorphoCLIPImageEncoder


def _make_encoder(aggregator: str = "ccf-mean") -> MorphoCLIPImageEncoder:
    return MorphoCLIPImageEncoder(
        embed_dim=64,
        output_dim=32,
        aggregator=aggregator,
        ccf_layers=1,
        ccf_heads=4,
        input_channels=5,
        proj_hidden_dim=32,
        proj_dropout=0.0,
    )


class TestMorphoCLIPImageEncoder:
    """Test suite for the full image encoder pipeline."""

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_output_shape(self, aggregator: str) -> None:
        encoder = _make_encoder(aggregator)
        features = torch.randn(2, 4, 5, 64)  # 2 wells, 4 sites
        site_mask = torch.ones(2, 4, dtype=torch.bool)
        out = encoder(features, site_mask)
        assert out.shape == (2, 32)

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_l2_normalized(self, aggregator: str) -> None:
        encoder = _make_encoder(aggregator)
        features = torch.randn(3, 2, 5, 64)
        site_mask = torch.ones(3, 2, dtype=torch.bool)
        out = encoder(features, site_mask)
        norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=0)

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_masking_ignores_padding(self, aggregator: str) -> None:
        """Padded sites should not affect the output."""
        encoder = _make_encoder(aggregator)
        encoder.eval()

        # Well with 2 real sites
        features_2 = torch.randn(1, 2, 5, 64)
        mask_2 = torch.tensor([[True, True]])

        # Same well padded to 4 sites
        features_4 = torch.zeros(1, 4, 5, 64)
        features_4[:, :2] = features_2
        mask_4 = torch.tensor([[True, True, False, False]])

        out_2 = encoder(features_2, mask_2)
        out_4 = encoder(features_4, mask_4)
        torch.testing.assert_close(out_2, out_4, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_variable_sites(self, aggregator: str) -> None:
        """Handles wells with different site counts via masking."""
        encoder = _make_encoder(aggregator)
        features = torch.randn(2, 5, 5, 64)  # padded to 5 sites
        site_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        out = encoder(features, site_mask)
        assert out.shape == (2, 32)

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_single_site(self, aggregator: str) -> None:
        encoder = _make_encoder(aggregator)
        features = torch.randn(1, 1, 5, 64)
        site_mask = torch.tensor([[True]])
        out = encoder(features, site_mask)
        assert out.shape == (1, 32)

    def test_unknown_aggregator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregator"):
            _make_encoder("not-an-aggregator")

    def test_default_aggregator_is_ccf_mean(self) -> None:
        encoder = MorphoCLIPImageEncoder(embed_dim=64, output_dim=32, ccf_layers=1, ccf_heads=4)
        assert encoder.aggregator == "ccf-mean"

    def test_ccf_mean_matches_reference_implementation(self) -> None:
        """``ccf-mean`` must reproduce the pre-refactor default path: CCF over
        channels, masked mean over sites, then ProjectionHead."""
        encoder = _make_encoder("ccf-mean")
        encoder.eval()

        features = torch.randn(3, 4, 5, 64)
        site_mask = torch.tensor(
            [
                [True, True, True, True],
                [True, True, False, False],
                [True, False, False, False],
            ]
        )

        B, S, C, D = features.shape
        x = features.reshape(B * S, C, D)
        x = encoder.cross_channel_former(x)
        x = x.view(B, S, D)
        mask = site_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        expected = encoder.projection(x)

        torch.testing.assert_close(encoder(features, site_mask), expected, atol=0.0, rtol=0.0)

    def test_meanpool_mean_matches_reference_implementation(self) -> None:
        """``meanpool-mean`` must reproduce the pre-refactor ``mean_pool`` path."""
        encoder = _make_encoder("meanpool-mean")
        encoder.eval()

        features = torch.randn(2, 3, 5, 64)
        site_mask = torch.tensor([[True, True, True], [True, True, False]])

        x = F.normalize(features, dim=-1).mean(dim=2)  # (B, S, D)
        mask = site_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        expected = encoder.projection(x)

        torch.testing.assert_close(encoder(features, site_mask), expected, atol=1e-6, rtol=1e-6)

    def test_submodules_per_aggregator(self) -> None:
        assert hasattr(_make_encoder("ccf-mean"), "cross_channel_former")
        assert not hasattr(_make_encoder("meanpool-mean"), "cross_channel_former")
        assert hasattr(_make_encoder("ccf-attn"), "site_pooling")
        assert not hasattr(_make_encoder("ccf-mean"), "site_pooling")
        assert hasattr(_make_encoder("wellformer"), "well_former")
