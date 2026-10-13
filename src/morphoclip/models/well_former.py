"""WellFormer: joint transformer over all site-channel tokens of a well.

The ``ccf-*`` aggregators compress channels per site and only then pool sites.
WellFormer attends over every ``(site, channel)`` token at once, so one
well-CLS token weights channels and sites together. Token count is ``1 + S * C``
(81 at S=16, C=5), so full attention stays cheap.

There is no site or positional embedding: sites are unordered, and the
aggregator must be permutation invariant over them.
"""

import torch

from morphoclip.models.channel_transformer import ChannelTokenTransformer


class WellFormer(ChannelTokenTransformer):
    """Transformer aggregating all site-channel tokens of a well into one vector.

    A learnable well-CLS token is prepended to the flattened ``S * C``
    site-channel tokens; its output is the well representation. Learnable
    channel-type embeddings, shared across sites, let the transformer tell the
    5 fluorescence channels apart.

    Constructor arguments are those of
    :class:`~morphoclip.models.channel_transformer.ChannelTokenTransformer`.
    """

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
        self._check_shape(channels, dim)
        if site_mask.shape != (batch_size, num_sites):
            raise ValueError(
                f"Expected site_mask of shape {(batch_size, num_sites)}, "
                f"got {tuple(site_mask.shape)}"
            )

        tokens = self._embed_channels(x).reshape(batch_size, num_sites * channels, dim)

        # A padded site masks all of its channel tokens.
        token_padding = ~site_mask.unsqueeze(-1).expand(batch_size, num_sites, channels)
        token_padding = token_padding.reshape(batch_size, num_sites * channels)

        return self._encode(tokens, token_padding)
