"""Gated-attention pooling over the sites of a well.

Sites in a well are an unordered bag, and some are out of focus, empty, or
off-target. Masked mean pooling weights them all equally; gated attention
(Ilse et al., 2018, "Attention-based Deep Multiple Instance Learning") learns
a per-site weight instead.
"""

import torch
import torch.nn as nn

ATTENTION_HIDDEN_DIM = 256


class AttentionSitePooling(nn.Module):
    """Gated-attention pooling over the site axis.

    Computes ``a_i = w2 @ (tanh(W1 h_i) * sigmoid(Wg h_i))`` for each site,
    softmaxes the scores over real (unpadded) sites, and returns the weighted
    sum of site representations.

    Args:
        embed_dim: Site representation dimension.
        hidden_dim: Attention bottleneck dimension.
        dropout: Dropout applied to the gated attention features.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        hidden_dim: int = ATTENTION_HIDDEN_DIM,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention_v = nn.Linear(embed_dim, hidden_dim)
        self.attention_gate = nn.Linear(embed_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def compute_weights(self, x: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        """Return per-site attention weights.

        Args:
            x: Site representations ``(B, S, D)``.
            site_mask: Boolean mask ``(B, S)``, ``True`` for real sites.

        Returns:
            Attention weights ``(B, S)`` summing to 1 over real sites (all
            zero for rows with no real site).

        Raises:
            ValueError: If the feature dimension does not match ``embed_dim``.
        """
        if x.shape[-1] != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {x.shape[-1]}")

        gated = torch.tanh(self.attention_v(x)) * torch.sigmoid(self.attention_gate(x))
        scores = self.attention_score(self.dropout(gated)).squeeze(-1)  # (B, S)

        scores = scores.masked_fill(~site_mask, float("-inf"))
        # A fully padded row would softmax to NaN, so flatten it and zero its
        # weights below. Wells always have >=1 site in practice.
        empty = ~site_mask.any(dim=1, keepdim=True)  # (B, 1)
        scores = torch.where(empty.expand_as(scores), torch.zeros_like(scores), scores)

        weights = torch.softmax(scores, dim=1)
        return weights.masked_fill(empty, 0.0)

    def forward(self, x: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        """Pool site representations into one vector per well.

        Args:
            x: Site representations ``(B, S, D)``.
            site_mask: Boolean mask ``(B, S)``, ``True`` for real sites.

        Returns:
            Pooled well representations ``(B, D)``.
        """
        weights = self.compute_weights(x, site_mask)
        return (weights.unsqueeze(-1) * x).sum(dim=1)
