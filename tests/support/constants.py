"""Shapes the fake feature builders and the tests that read them must agree on."""

from morphoclip.training.config import INPUT_CHANNELS

NUM_CHANNELS = INPUT_CHANNELS

# Small stand-in for EMBED_DIM. Anything that builds a real image encoder must
# use EMBED_DIM instead, since that width is baked into the module.
HIDDEN_DIM = 16
