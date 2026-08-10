"""Stand-ins for the HuggingFace pieces, so tests never download a model."""

from collections.abc import Sequence
from types import SimpleNamespace

import torch


class DummyTokenizer:
    """Tokenizer stub returning ``max_length`` ids per text.

    The ids are ``arange``, not real tokens: collate tests care about shapes and
    about which texts reach the tokenizer, never about the vocabulary.
    """

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool | str,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        del padding, truncation, return_tensors
        return {
            "input_ids": torch.arange(len(texts) * max_length).reshape(len(texts), max_length),
            "attention_mask": torch.ones(len(texts), max_length, dtype=torch.long),
        }


class FakeTextModel(torch.nn.Module):
    """BERT-like stub exposing the ``pooler_output`` the encoders read."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(4096, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(pooler_output=self.embedding(input_ids).mean(dim=1))
