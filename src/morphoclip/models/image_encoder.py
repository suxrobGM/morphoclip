"""Full image encoder: site features -> contrastive space embedding.

Wraps channel aggregation (CrossChannelFormer or mean pooling), masked
mean pooling (site-to-well aggregation), and ProjectionHead (1024 -> 512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphoclip.models.cross_channel_former import CrossChannelFormer
from morphoclip.models.projection_head import ProjectionHead


class MorphoCLIPImageEncoder(nn.Module):
    """Image encoder mapping pre-extracted DINOv3 features to contrastive space.

    Pipeline::

        (B, max_sites, 5, 1024)  -- per-site channel CLS tokens
            -> channel aggregation -- aggregate 5 channels into 1 per site
        (B, max_sites, 1024)
            -> masked mean pool    -- aggregate sites into 1 per well
        (B, 1024)
            -> ProjectionHead      -- project to 512-d L2-normalized
        (B, 512)

    Args:
        embed_dim: DINOv3 feature dimension.
        output_dim: Contrastive space dimension.
        channel_aggregation: ``"ccf"`` for CrossChannelFormer or
            ``"mean_pool"`` for simple L2-normalized mean pooling.
        ccf_layers: CrossChannelFormer transformer layers.
        ccf_heads: CrossChannelFormer attention heads.
        input_channels: Number of fluorescence channels.
        proj_hidden_dim: ProjectionHead hidden dimension.
        proj_dropout: ProjectionHead dropout rate.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        output_dim: int = 512,
        channel_aggregation: str = "ccf",
        ccf_layers: int = 2,
        ccf_heads: int = 8,
        input_channels: int = 5,
        proj_hidden_dim: int = 512,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_aggregation = channel_aggregation

        if channel_aggregation == "ccf":
            self.cross_channel_former = CrossChannelFormer(
                embed_dim=embed_dim,
                num_layers=ccf_layers,
                num_heads=ccf_heads,
                input_channels=input_channels,
            )
        elif channel_aggregation != "mean_pool":
            raise ValueError(
                f"Unknown channel_aggregation: {channel_aggregation!r}. Use 'ccf' or 'mean_pool'."
            )

        self.projection = ProjectionHead(
            input_dim=embed_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=output_dim,
            dropout=proj_dropout,
        )

    def _aggregate_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate channel tokens into one vector per site.

        Args:
            x: ``(B_sites, C, D)`` per-channel CLS tokens.

        Returns:
            ``(B_sites, D)`` aggregated representation.
        """
        if self.channel_aggregation == "ccf":
            return self.cross_channel_former(x)
        # mean_pool: L2-normalize each channel, then average
        x = F.normalize(x, dim=-1)
        return x.mean(dim=1)

    def forward(
        self,
        features: torch.Tensor,
        site_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode well features to contrastive space.

        Args:
            features: Padded site features
                ``(B, max_sites, num_channels, embed_dim)``.
            site_mask: Boolean mask ``(B, max_sites)``, ``True`` for real
                sites, ``False`` for padding.

        Returns:
            Well embeddings ``(B, output_dim)``, L2-normalized.
        """
        B, S, C, D = features.shape

        # Per-site channel aggregation
        x = features.reshape(B * S, C, D)  # (B*S, C, D)
        x = self._aggregate_channels(x)  # (B*S, D)
        x = x.view(B, S, D)  # (B, S, D)

        # Masked mean pooling over sites
        mask = site_mask.unsqueeze(-1).float()  # (B, S, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, D)

        # Project to contrastive space
        x = self.projection(x)  # (B, output_dim), L2-normalized
        return x
