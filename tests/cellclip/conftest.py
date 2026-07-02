"""Shared fixtures for CellCLIP tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from morphoclip.data.metadata import MetadataIndex

NUM_CHANNELS = 5
HIDDEN_DIM = 16


class DummyTokenizer:
    """Small tokenizer stub for collate tests."""

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        del padding, truncation, return_tensors
        batch_size = len(texts)
        input_ids = torch.arange(batch_size * max_length).reshape(batch_size, max_length)
        attention_mask = torch.ones(batch_size, max_length, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeTextModel(torch.nn.Module):
    """BERT-like stub with a pooler output."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(4096, hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        pooled = self.embedding(input_ids).mean(dim=1)
        return SimpleNamespace(pooler_output=pooled)


@pytest.fixture
def metadata_index(metadata_dir: Path) -> MetadataIndex:
    return MetadataIndex.from_directory(metadata_dir, batch="2020_11_04_CPJUMP1")


def write_feature(feature_root: Path, plate: str, well: str, *, sites: int = 1) -> None:
    """Write fake feature files for testing."""
    row = ord(well[0].upper()) - ord("A") + 1
    col = int(well[1:])
    plate_dir = feature_root / plate
    plate_dir.mkdir(parents=True, exist_ok=True)
    for site in range(1, sites + 1):
        torch.save(
            torch.randn(NUM_CHANNELS, HIDDEN_DIM),
            plate_dir / f"r{row:02d}c{col:02d}f{site:02d}.pt",
        )
