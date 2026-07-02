"""Tests for MorphoCLIPImageEncoder."""

import torch

from morphoclip.models.image_encoder import MorphoCLIPImageEncoder


class TestMorphoCLIPImageEncoder:
    """Test suite for the full image encoder pipeline."""

    def _make_encoder(self) -> MorphoCLIPImageEncoder:
        return MorphoCLIPImageEncoder(
            embed_dim=64,
            output_dim=32,
            ccf_layers=1,
            ccf_heads=4,
            input_channels=5,
            proj_hidden_dim=32,
            proj_dropout=0.0,
        )

    def test_output_shape(self) -> None:
        encoder = self._make_encoder()
        features = torch.randn(2, 4, 5, 64)  # 2 wells, 4 sites
        site_mask = torch.ones(2, 4, dtype=torch.bool)
        out = encoder(features, site_mask)
        assert out.shape == (2, 32)

    def test_l2_normalized(self) -> None:
        encoder = self._make_encoder()
        features = torch.randn(3, 2, 5, 64)
        site_mask = torch.ones(3, 2, dtype=torch.bool)
        out = encoder(features, site_mask)
        norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=0)

    def test_masking_ignores_padding(self) -> None:
        """Padded sites should not affect the output."""
        encoder = self._make_encoder()
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

    def test_variable_sites(self) -> None:
        """Handles wells with different site counts via masking."""
        encoder = self._make_encoder()
        features = torch.randn(2, 5, 5, 64)  # padded to 5 sites
        site_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        out = encoder(features, site_mask)
        assert out.shape == (2, 32)

    def test_single_site(self) -> None:
        encoder = self._make_encoder()
        features = torch.randn(1, 1, 5, 64)
        site_mask = torch.tensor([[True]])
        out = encoder(features, site_mask)
        assert out.shape == (1, 32)
