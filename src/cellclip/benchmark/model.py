"""Minimal image-only CellCLIP implementation for profile export."""

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CellCLIPVisualConfig:
    """Visual encoder settings for the published CellCLIP checkpoint."""

    embed_dim: int = 512
    vision_layers: int = 12
    vision_width: int = 1536
    vision_heads: int = 8
    input_channels: int = 5
    pooling: str = "attention"


class LayerNorm(nn.LayerNorm):
    """LayerNorm variant that is safe under mixed precision."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        ret = super().forward(x.float())
        return ret.to(orig_type)


class QuickGELU(nn.Module):
    """OpenAI CLIP-style GELU approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """Transformer residual block used by CellCLIP's channel encoder."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """Stack of residual attention blocks."""

    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class CrossChannelFormer(nn.Module):
    """Transformer over per-channel site embeddings."""

    def __init__(
        self,
        embed_dim: int,
        layers: int,
        heads: int,
        output_dim: int,
        input_channels: int,
        use_cls_token: bool = True,
    ):
        super().__init__()
        scale = embed_dim**-0.5

        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.use_cls_token = use_cls_token

        self.channel_embed = nn.Parameter(torch.zeros(input_channels, embed_dim))
        nn.init.trunc_normal_(self.channel_embed, std=0.02)
        self.channel_ln = LayerNorm(embed_dim)
        self.ln_pre = LayerNorm(embed_dim)
        self.transformer = Transformer(embed_dim, layers, heads)
        self.cls_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.ln_post = LayerNorm(embed_dim)
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, width = x.shape
        if channels != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, got {channels}")
        if width != self.embed_dim:
            raise ValueError(f"Expected input width {self.embed_dim}, got {width}")

        x = x + self.channel_embed.unsqueeze(0).to(x.dtype)
        x = self.channel_ln(x)
        cls_tokens = self.cls_token.unsqueeze(1).expand(batch_size, -1, -1).to(x.dtype)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        if self.use_cls_token:
            x = self.ln_post(x[:, 0, :])
        else:
            x = self.ln_post(x[:, 1:, :].mean(dim=1))

        if self.embed_dim != self.output_dim:
            x = x @ self.proj
        return x


class MILPooling(nn.Module):
    """Channel-independent multi-instance pooling used before the visual encoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        if pooling == "attention":
            self.V = nn.Linear(input_dim, hidden_dim)
            self.U = nn.Linear(input_dim, hidden_dim)
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool site bags from ``(B, M, C, D)`` to ``(B, C, D)``."""
        batch_size, num_sites, num_channels, width = x.shape

        if self.pooling == "attention":
            x = x.permute(0, 2, 1, 3).reshape(batch_size * num_channels, num_sites, width)
            h_v = torch.tanh(self.V(x))
            h_u = torch.sigmoid(self.U(x))
            attn_scores = self.attention(h_v * h_u)
            mask = (x.abs().sum(dim=-1) > 0).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = torch.sum(attn_weights * x, dim=1)
            return pooled.view(batch_size, num_channels, width)

        if self.pooling == "median":
            return torch.median(x, dim=1).values

        return torch.mean(x, dim=1)


class CellCLIPVisualEncoder(nn.Module):
    """Image-only CellCLIP wrapper exposing ``encode_image``."""

    def __init__(self, config: CellCLIPVisualConfig):
        super().__init__()
        self.config = config
        self.image_pool = MILPooling(
            input_dim=config.vision_width,
            pooling=config.pooling,
        )
        self.visual = CrossChannelFormer(
            embed_dim=config.vision_width,
            layers=config.vision_layers,
            heads=config.vision_heads,
            output_dim=config.embed_dim,
            input_channels=config.input_channels,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def encode_mil(self, image: torch.Tensor) -> torch.Tensor:
        """Pool site bags before the visual transformer."""
        return self.image_pool(image)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return self.visual(image).to(self.dtype)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 4:
            image = self.encode_mil(image)
        return self.encode_image(image)
