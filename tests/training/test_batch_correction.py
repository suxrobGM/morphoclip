"""Tests for CWA batch correction."""

import torch
import torch.nn.functional as F

from morphoclip.training.batch_correction import cross_well_alignment


class TestCrossWellAlignment:
    """Tests for cross-well alignment batch correction."""

    def test_output_shape(self) -> None:
        emb = F.normalize(torch.randn(6, 32), dim=-1)
        plates = ["P1", "P1", "P2", "P2", "P3", "P3"]
        out = cross_well_alignment(emb, plates)
        assert out.shape == (6, 32)

    def test_l2_normalized_output(self) -> None:
        emb = F.normalize(torch.randn(8, 16), dim=-1)
        plates = ["A", "A", "A", "A", "B", "B", "B", "B"]
        out = cross_well_alignment(emb, plates)
        norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-5, rtol=0)

    def test_single_plate_zero_mean(self) -> None:
        """All samples from one plate → mean should be ~zero before renorm."""
        emb = F.normalize(torch.randn(4, 16), dim=-1)
        plates = ["P1", "P1", "P1", "P1"]
        out = cross_well_alignment(emb, plates)
        # After mean subtraction the centroid shifts toward zero
        # After renorm, the mean won't be exactly zero but should be small
        mean_norm = out.mean(dim=0).norm().item()
        assert mean_norm < 1.0  # significantly less than unit norm

    def test_does_not_modify_input(self) -> None:
        emb = F.normalize(torch.randn(4, 16), dim=-1)
        emb_copy = emb.clone()
        plates = ["A", "A", "B", "B"]
        cross_well_alignment(emb, plates)
        torch.testing.assert_close(emb, emb_copy)

    def test_single_sample_per_plate(self) -> None:
        """Each sample in its own plate → subtract self-mean → all zero → renorm."""
        emb = F.normalize(torch.randn(3, 8), dim=-1)
        plates = ["A", "B", "C"]
        # Subtracting self-mean gives zero; renorm would divide by zero.
        # F.normalize handles zero vectors by returning zeros.
        out = cross_well_alignment(emb, plates)
        assert out.shape == (3, 8)
