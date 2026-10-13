"""CrossChannelFormer: transformer over the per-channel DINOv3 CLS tokens of one site.

Lets the 5 fluorescence channels (Mitochondria, Actin, Golgi, ER, DNA) attend
to each other and returns one vector per site.
"""

import torch

from morphoclip.models.channel_transformer import ChannelTokenTransformer


class CrossChannelFormer(ChannelTokenTransformer):
    """2-layer transformer aggregating 5 channel CLS tokens into 1 representation.

    A learnable CLS token is prepended to the 5 channel tokens; its output is
    the aggregated representation. Learnable channel-type embeddings let the
    transformer tell the channels apart.

    This is not CellCLIP's CrossChannelFormer (1536-d, 12 layers, built-in
    projection). It takes DINOv3 1024-d features, uses 2 layers, and has no
    output projection; the downstream ``ProjectionHead`` maps to the
    contrastive space.

    Constructor arguments are those of
    :class:`~morphoclip.models.channel_transformer.ChannelTokenTransformer`.
    """

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
        _, channels, dim = x.shape
        self._check_shape(channels, dim)
        return self._encode(self._embed_channels(x))
