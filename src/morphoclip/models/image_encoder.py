"""Full image encoder: site features -> contrastive space embedding.

Wraps well aggregation (channel + site pooling, selected by ``aggregator``)
and ProjectionHead (1024 -> 512).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from morphoclip.models.cross_channel_former import CrossChannelFormer
from morphoclip.models.projection_head import ProjectionHead
from morphoclip.models.site_pooling import AttentionSitePooling
from morphoclip.models.well_former import WellFormer

# Channel-then-site aggregators, named "<channel step>-<site step>", plus the
# joint "wellformer" which does both steps in one attention pass.
AGGREGATORS = ("ccf-mean", "meanpool-mean", "ccf-attn", "wellformer")


class MorphoCLIPImageEncoder(nn.Module):
    """Image encoder mapping pre-extracted DINOv3 features to contrastive space.

    Pipeline::

        (B, max_sites, 5, 1024)  -- per-site channel CLS tokens
            -> well aggregation    -- channels + sites -> 1 vector per well
        (B, 1024)
            -> ProjectionHead      -- project to 512-d L2-normalized
        (B, 512)

    Args:
        embed_dim: DINOv3 feature dimension.
        output_dim: Contrastive space dimension.
        aggregator: Well aggregation strategy. ``"ccf-mean"`` (default)
            runs the CrossChannelFormer over channels then masked mean over
            sites; ``"meanpool-mean"`` replaces the CCF with L2-normalized
            channel averaging; ``"ccf-attn"`` replaces the site mean with
            gated-attention MIL pooling; ``"wellformer"`` attends over all
            site-channel tokens jointly.
        ccf_layers: Transformer layers (CrossChannelFormer or WellFormer).
        ccf_heads: Attention heads (CrossChannelFormer or WellFormer).
        input_channels: Number of fluorescence channels.
        proj_hidden_dim: ProjectionHead hidden dimension.
        proj_dropout: ProjectionHead dropout rate.

    Raises:
        ValueError: If *aggregator* is not one of :data:`AGGREGATORS`.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        output_dim: int = 512,
        aggregator: str = "ccf-mean",
        ccf_layers: int = 2,
        ccf_heads: int = 8,
        input_channels: int = 5,
        proj_hidden_dim: int = 512,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if aggregator not in AGGREGATORS:
            raise ValueError(
                f"Unknown aggregator: {aggregator!r}. Use one of {', '.join(AGGREGATORS)}."
            )
        self.aggregator = aggregator

        if aggregator in ("ccf-mean", "ccf-attn"):
            self.cross_channel_former = CrossChannelFormer(
                embed_dim=embed_dim,
                num_layers=ccf_layers,
                num_heads=ccf_heads,
                input_channels=input_channels,
            )
        elif aggregator == "wellformer":
            self.well_former = WellFormer(
                embed_dim=embed_dim,
                num_layers=ccf_layers,
                num_heads=ccf_heads,
                input_channels=input_channels,
            )

        if aggregator == "ccf-attn":
            self.site_pooling = AttentionSitePooling(embed_dim=embed_dim)

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
        if self.aggregator in ("ccf-mean", "ccf-attn"):
            return self.cross_channel_former(x)
        # meanpool: L2-normalize each channel, then average
        x = F.normalize(x, dim=-1)
        return x.mean(dim=1)

    def _aggregate_sites(self, x: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate per-site representations into one vector per well.

        Args:
            x: ``(B, S, D)`` per-site representations.
            site_mask: Boolean mask ``(B, S)``, ``True`` for real sites.

        Returns:
            ``(B, D)`` well representation.
        """
        if self.aggregator == "ccf-attn":
            return self.site_pooling(x, site_mask)
        mask = site_mask.unsqueeze(-1).float()  # (B, S, 1)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    def _aggregate_well(self, features: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        """Reduce padded site features to one vector per well."""
        if self.aggregator == "wellformer":
            return self.well_former(features, site_mask)

        B, S, C, D = features.shape
        x = features.reshape(B * S, C, D)  # (B*S, C, D)
        x = self._aggregate_channels(x)  # (B*S, D)
        x = x.view(B, S, D)  # (B, S, D)
        return self._aggregate_sites(x, site_mask)  # (B, D)

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
        x = self._aggregate_well(features, site_mask)  # (B, D)
        return self.projection(x)  # (B, output_dim), L2-normalized
