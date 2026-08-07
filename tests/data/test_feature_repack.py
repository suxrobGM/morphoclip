"""Tests for cached-feature repacking."""

import torch

from morphoclip.cli.features import repack_feature_file


def _write_sliced_feature(path, n_sites: int = 8, channels: int = 5, dim: int = 64):
    """Save one site as a bare slice of a batch tensor (the oversized form)."""
    batch = torch.randn(n_sites, channels, dim)
    torch.save(batch[0], path)
    return batch[0].clone()


class TestRepackFeatureFile:
    def test_shrinks_oversized_file(self, tmp_path):
        path = tmp_path / "r01c01f01.pt"
        _write_sliced_feature(path)

        before, after = repack_feature_file(path)

        assert after < before

    def test_preserves_values_exactly(self, tmp_path):
        path = tmp_path / "r01c01f01.pt"
        expected = _write_sliced_feature(path)

        repack_feature_file(path)

        assert torch.equal(torch.load(path, map_location="cpu"), expected)

    def test_drops_excess_storage(self, tmp_path):
        path = tmp_path / "r01c01f01.pt"
        _write_sliced_feature(path)

        repack_feature_file(path)

        tensor = torch.load(path, map_location="cpu")
        assert tensor.untyped_storage().nbytes() == tensor.nbytes

    def test_already_packed_file_is_untouched(self, tmp_path):
        path = tmp_path / "r01c01f01.pt"
        torch.save(torch.randn(5, 64), path)
        original_size = path.stat().st_size

        before, after = repack_feature_file(path)

        assert (before, after) == (original_size, original_size)
        assert path.stat().st_size == original_size

    def test_leaves_no_temp_file(self, tmp_path):
        path = tmp_path / "r01c01f01.pt"
        _write_sliced_feature(path)

        repack_feature_file(path)

        assert list(tmp_path.glob("*.tmp")) == []
