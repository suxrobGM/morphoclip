"""Stand-ins for the HuggingFace pieces, so tests never download a model."""

from types import SimpleNamespace

import torch


class FakeTextModel(torch.nn.Module):
    """BERT-like stub exposing the ``pooler_output`` the encoders read."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(4096, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(pooler_output=self.embedding(input_ids).mean(dim=1))
