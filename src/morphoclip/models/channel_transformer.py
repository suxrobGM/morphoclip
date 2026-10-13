"""Shared body of the transformer aggregators over per-channel DINOv3 CLS tokens.

:class:`~morphoclip.models.cross_channel_former.CrossChannelFormer` and
:class:`~morphoclip.models.well_former.WellFormer` differ only in which tokens
they attend over. Both keep the parameter names defined here, which are the
checkpoint state-dict keys.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelTokenTransformer(nn.Module):
    """CLS-token transformer over channel tokens carrying a learned channel embedding.

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

        self.channel_embed = nn.Parameter(torch.zeros(input_channels, embed_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

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

    def _check_shape(self, channels: int, dim: int) -> None:
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        if dim != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {dim}")

    def _embed_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Add the channel embedding to ``(..., C, D)`` features."""
        # Normalize raw DINOv3 features so the added channel embeddings are
        # on a comparable scale.
        return F.normalize(x, dim=-1) + self.channel_embed

    def _encode(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Prepend the CLS token to ``(B, N, D)`` tokens and return its output ``(B, D)``.

        *key_padding_mask*, when given, covers the ``N`` tokens only; the CLS
        token is never masked.
        """
        batch_size = tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1 + N, D)
        if key_padding_mask is not None:
            cls_padding = torch.zeros(
                (batch_size, 1), dtype=key_padding_mask.dtype, device=key_padding_mask.device
            )
            key_padding_mask = torch.cat([cls_padding, key_padding_mask], dim=1)

        tokens = self.ln_pre(tokens)
        tokens = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        return self.ln_post(tokens[:, 0, :])  # (B, D)
