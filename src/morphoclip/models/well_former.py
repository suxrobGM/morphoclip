"""WellFormer: joint transformer over all site-channel tokens of a well.

Where the ``ccf-*`` aggregators compress channels per site and only then pool
sites, WellFormer attends over every ``(site, channel)`` token at once, so a
single well-CLS token can weight channels and sites jointly.  Token count is
``1 + S * C`` (81 at S=16, C=5), so full attention stays cheap.

No site/positional embedding is used: sites within a well are unordered, and
permutation invariance over sites is a required property.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WellFormer(nn.Module):
    """Transformer aggregating all site-channel tokens of a well into one vector.

    A learnable well-CLS token is prepended to the flattened ``S * C``
    site-channel tokens, and its output is the well representation.  Learnable
    channel-type embeddings (shared across sites) let the transformer tell the
    5 fluorescence channels apart.

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

        # Learnable channel-type embeddings (Mito, Actin, Golgi, ER, DNA),
        # broadcast across sites.  Deliberately no site embedding.
        self.channel_embed = nn.Parameter(torch.zeros(input_channels, embed_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)

        # Learnable well-level CLS token for aggregation
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

    def forward(self, x: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        """Aggregate a well's site-channel tokens into a single representation.

        Args:
            x: Padded site features ``(B, S, C, D)``.
            site_mask: Boolean mask ``(B, S)``, ``True`` for real sites.

        Returns:
            Well representations ``(B, D)``.

        Raises:
            ValueError: If the channel or dimension sizes are unexpected, or
                if *site_mask* does not match the site axis of *x*.
        """
        batch_size, num_sites, channels, dim = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        if dim != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {dim}")
        if site_mask.shape != (batch_size, num_sites):
            raise ValueError(
                f"Expected site_mask of shape {(batch_size, num_sites)}, "
                f"got {tuple(site_mask.shape)}"
            )

        # L2-normalize raw DINOv3 features so channel embeddings are meaningful
        x = F.normalize(x, dim=-1)

        # Add channel-type embeddings (broadcast over sites)
        x = x + self.channel_embed.view(1, 1, channels, dim)

        # Flatten site-channel tokens and prepend the well-CLS token
        tokens = x.reshape(batch_size, num_sites * channels, dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 1 + S*C, D)

        # Padded sites mask all of their channel tokens; CLS is never masked
        token_padding = ~site_mask.unsqueeze(-1).expand(batch_size, num_sites, channels)
        token_padding = token_padding.reshape(batch_size, num_sites * channels)
        cls_padding = torch.zeros(
            (batch_size, 1), dtype=token_padding.dtype, device=token_padding.device
        )
        key_padding_mask = torch.cat([cls_padding, token_padding], dim=1)

        tokens = self.ln_pre(tokens)
        tokens = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        return self.ln_post(tokens[:, 0, :])  # (B, D)
