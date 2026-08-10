"""Tests for cached-feature repacking."""

from pathlib import Path

import torch

from morphoclip.data.feature_extractor import repack_feature_file


def _write_sliced_feature(path: Path) -> torch.Tensor:
    """Save one site as a bare slice of a batch tensor (the oversized form)."""
    batch = torch.randn(8, 5, 64)
    torch.save(batch[0], path)
    return batch[0].clone()


def test_repacking_drops_the_shared_storage_and_keeps_the_values(tmp_path: Path) -> None:
    path = tmp_path / "r01c01f01.pt"
    expected = _write_sliced_feature(path)

    before, after = repack_feature_file(path)

    assert after < before
    assert after == path.stat().st_size
    assert list(tmp_path.glob("*.tmp")) == []

    tensor = torch.load(path, map_location="cpu")
    assert torch.equal(tensor, expected)
    assert tensor.untyped_storage().nbytes() == tensor.nbytes


def test_an_already_packed_file_is_left_on_disk_untouched(tmp_path: Path) -> None:
    path = tmp_path / "r01c01f01.pt"
    torch.save(torch.randn(5, 64), path)
    original_size = path.stat().st_size

    assert repack_feature_file(path) == (original_size, original_size)
    assert path.stat().st_size == original_size
