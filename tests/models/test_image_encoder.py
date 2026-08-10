"""Tests for MorphoCLIPImageEncoder."""

import pytest
import torch

from morphoclip.models.image_encoder import AGGREGATORS, MorphoCLIPImageEncoder


def _make_encoder(aggregator: str = "ccf-mean") -> MorphoCLIPImageEncoder:
    encoder = MorphoCLIPImageEncoder(
        embed_dim=64,
        output_dim=32,
        aggregator=aggregator,
        ccf_layers=1,
        ccf_heads=4,
        input_channels=5,
        proj_hidden_dim=32,
        proj_dropout=0.0,
    )
    encoder.eval()
    return encoder


class TestMorphoCLIPImageEncoder:
    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_output_is_one_unit_norm_row_per_well(self, aggregator: str) -> None:
        out = _make_encoder(aggregator)(
            torch.randn(3, 4, 5, 64), torch.ones(3, 4, dtype=torch.bool)
        )
        assert out.shape == (3, 32)
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=0)

    def test_unknown_aggregator_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown aggregator"):
            _make_encoder("not-an-aggregator")

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_masked_sites_are_ignored_even_when_they_hold_garbage(self, aggregator: str) -> None:
        """Padding with noise, not zeros, is what actually pins the masking.

        Zero padding passes even when the mask is applied to the numerator only,
        because the zeros contribute nothing to the sum either way. Noise does not.
        """
        encoder = _make_encoder(aggregator)
        real = torch.randn(1, 2, 5, 64)
        padded = torch.randn(1, 4, 5, 64)
        padded[:, :2] = real

        torch.testing.assert_close(
            encoder(real, torch.tensor([[True, True]])),
            encoder(padded, torch.tensor([[True, True, False, False]])),
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.parametrize("aggregator", AGGREGATORS)
    def test_site_order_does_not_matter(self, aggregator: str) -> None:
        """Sites within a well are an unordered bag; no aggregator may encode order."""
        encoder = _make_encoder(aggregator)
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
        one_site = torch.randn(1, 1, 5, 64)

        torch.testing.assert_close(
            encoder(one_site, torch.ones(1, 1, dtype=torch.bool)),
            encoder(one_site.repeat(1, 2, 1, 1), torch.ones(1, 2, dtype=torch.bool)),
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
        permutation = torch.tensor([3, 1, 4, 0, 2])

        meanpool = _make_encoder("meanpool-mean")
        torch.testing.assert_close(
            meanpool(features, site_mask),
            meanpool(features[:, :, permutation], site_mask),
            atol=1e-5,
            rtol=1e-5,
        )

        for aggregator in ("ccf-mean", "wellformer"):
            encoder = _make_encoder(aggregator)
            assert not torch.allclose(
                encoder(features, site_mask),
                encoder(features[:, :, permutation], site_mask),
                atol=1e-5,
            ), f"{aggregator} lost its per-channel embedding"
