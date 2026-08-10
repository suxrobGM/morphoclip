"""Tests for MorphoCLIPImageEncoder."""

import pytest
import torch

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

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_masked_sites_are_ignored_even_when_they_hold_garbage(self, aggregator: str) -> None:
        """Padding with noise, not zeros, is what actually pins the masking.

        Zero padding passes even when the mask is applied to the numerator only,
        because the zeros contribute nothing to the sum either way. Noise does not.
        """
        encoder = _make_encoder(aggregator)
        encoder.eval()

        real = torch.randn(1, 2, 5, 64)
        padded = torch.randn(1, 4, 5, 64)
        padded[:, :2] = real

        out_real = encoder(real, torch.tensor([[True, True]]))
        out_padded = encoder(padded, torch.tensor([[True, True, False, False]]))
        torch.testing.assert_close(out_real, out_padded, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_site_order_does_not_matter(self, aggregator: str) -> None:
        """Sites within a well are an unordered bag; no aggregator may encode order."""
        encoder = _make_encoder(aggregator)
        encoder.eval()

        features = torch.randn(1, 4, 5, 64)
        site_mask = torch.ones(1, 4, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 3, 1])

        torch.testing.assert_close(
            encoder(features, site_mask),
            encoder(features[:, permutation], site_mask[:, permutation]),
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("aggregator", ["ccf-mean", "meanpool-mean"])
    def test_repeating_a_site_does_not_change_the_well(self, aggregator: str) -> None:
        """A mean over sites is insensitive to duplication; catches denominator errors."""
        encoder = _make_encoder(aggregator)
        encoder.eval()

        one_site = torch.randn(1, 1, 5, 64)
        twice = one_site.repeat(1, 2, 1, 1)

        torch.testing.assert_close(
            encoder(one_site, torch.ones(1, 1, dtype=torch.bool)),
            encoder(twice, torch.ones(1, 2, dtype=torch.bool)),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_meanpool_ignores_channel_order_but_ccf_does_not(self) -> None:
        """The defining difference between the aggregators, in one assertion.

        meanpool averages channels, so their order cannot matter. CCF and
        WellFormer add a learned per-channel embedding, so it must.
        """
        features = torch.randn(1, 3, 5, 64)
        site_mask = torch.ones(1, 3, dtype=torch.bool)
        channel_permutation = torch.tensor([3, 1, 4, 0, 2])

        meanpool = _make_encoder("meanpool-mean")
        meanpool.eval()
        torch.testing.assert_close(
            meanpool(features, site_mask),
            meanpool(features[:, :, channel_permutation], site_mask),
            atol=1e-5,
            rtol=1e-5,
        )

        for aggregator in ("ccf-mean", "wellformer"):
            encoder = _make_encoder(aggregator)
            encoder.eval()
            assert not torch.allclose(
                encoder(features, site_mask),
                encoder(features[:, :, channel_permutation], site_mask),
                atol=1e-5,
            ), f"{aggregator} lost its per-channel embedding"

    def test_submodules_per_aggregator(self) -> None:
        assert hasattr(_make_encoder("ccf-mean"), "cross_channel_former")
        assert not hasattr(_make_encoder("meanpool-mean"), "cross_channel_former")
        assert hasattr(_make_encoder("ccf-attn"), "site_pooling")
        assert not hasattr(_make_encoder("ccf-mean"), "site_pooling")
        assert hasattr(_make_encoder("wellformer"), "well_former")
