"""CrossChannelFormer: transformer aggregator for per-channel DINOv3 CLS tokens.

Learns cross-channel interactions between the 5 fluorescence channels
(Mitochondria, Actin, Golgi, ER, DNA) and produces a single unified
image representation per site.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossChannelFormer(nn.Module):
    """2-layer transformer aggregating 5 channel CLS tokens into 1 representation.

    A learnable CLS token is prepended to the 5 channel tokens, and its
    output is used as the aggregated representation.  Learnable channel-type
    embeddings are added so the transformer can distinguish channels.

    Unlike CellCLIP's CrossChannelFormer (1536-d, 12 layers, built-in
    projection), this version operates on DINOv3 1024-d features with a
    lightweight 2-layer encoder and no output projection — the downstream
    ``ProjectionHead`` handles mapping to the contrastive space.

    Args:
        embed_dim: Input/output feature dimension (DINOv3 CLS = 1024).
        num_layers: Number of transformer encoder layers.
        num_heads: Number of attention heads.
        input_channels: Number of fluorescence channels (5).
        dropout: Dropout rate for transformer layers.
        ff_dim_factor: Feed-forward hidden dimension multiplier.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        input_channels: int = 5,
        dropout: float = 0.1,
        ff_dim_factor: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_channels = input_channels

        # Learnable channel-type embeddings (Mito, Actin, Golgi, ER, DNA)
        self.channel_embed = nn.Parameter(torch.zeros(input_channels, embed_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # Learnable CLS token for aggregation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.ln_pre = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ff_dim_factor,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.ln_post = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate channel tokens into a single representation.

        Args:
            x: Per-channel CLS tokens ``(B, C, D)`` where C is the number
                of channels and D is ``embed_dim``.

        Returns:
            Aggregated representation ``(B, D)``.

        Raises:
            ValueError: If the channel or dimension sizes are unexpected.
        """
        batch_size, channels, dim = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        if dim != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {dim}")

        # L2-normalize raw DINOv3 features so channel embeddings are meaningful
        x = F.normalize(x, dim=-1)

        # Add channel-type embeddings
        x = x + self.channel_embed.unsqueeze(0)  # (B, C, D)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, C+1, D)

        x = self.ln_pre(x)
        x = self.transformer(x)  # (B, C+1, D)

        # Extract CLS token output
        x = self.ln_post(x[:, 0, :])  # (B, D)
        return x
