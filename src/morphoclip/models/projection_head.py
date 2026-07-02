"""Projection head for mapping encoder output to shared contrastive space."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """MLP projection from encoder hidden dim to shared contrastive space.

    Architecture: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> L2 norm
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.net(x)  # [B, output_dim]
        return F.normalize(projected, dim=-1)  # L2 normalize
