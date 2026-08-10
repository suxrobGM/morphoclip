"""Tests for CWA batch correction."""

import torch
import torch.nn.functional as F

from morphoclip.training.batch_correction import cross_well_alignment


class TestCrossWellAlignment:
    def test_output_stays_l2_normalized(self) -> None:
        emb = F.normalize(torch.randn(8, 16), dim=-1)
        out = cross_well_alignment(emb, ["A", "A", "A", "A", "B", "B", "B", "B"])
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(8), atol=1e-5, rtol=0)

    def test_does_not_modify_input(self) -> None:
        emb = F.normalize(torch.randn(4, 16), dim=-1)
        emb_copy = emb.clone()
        cross_well_alignment(emb, ["A", "A", "B", "B"])
        torch.testing.assert_close(emb, emb_copy)

    def test_singleton_plate_preserved_alongside_corrected_plate(self) -> None:
        """A lone well keeps its embedding while multi-well plates are corrected."""
        emb = F.normalize(torch.randn(5, 16), dim=-1)
        out = cross_well_alignment(emb, ["A", "A", "A", "A", "SOLO"])
        torch.testing.assert_close(out[4], emb[4], atol=1e-6, rtol=0)
        assert not torch.allclose(out[:4], emb[:4], atol=1e-4)
