"""Tests for the text projection head."""

import torch

from morphoclip.models.projection_head import ProjectionHead


class TestProjectionHead:
    def test_returns_unit_norm_rows_of_the_output_width(self) -> None:
        out = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=512)(torch.randn(4, 768))
        assert out.shape == (4, 512)
        torch.testing.assert_close(out.norm(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
