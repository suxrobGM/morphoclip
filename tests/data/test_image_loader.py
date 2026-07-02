"""Tests for morphoclip.data.image_loader module."""

from pathlib import Path

import numpy as np
import pytest
import torch

from morphoclip.data.image_loader import (
    CHANNEL_NAMES,
    DINO_INPUT_SIZE,
    FLUORESCENCE_CHANNELS,
    ImageKey,
    discover_sites,
    load_single_channel,
    load_site,
    load_site_as_tensor,
    parse_filename,
    prepare_channels_for_dino,
)


class TestParseFilename:
    def test_standard_filename(self) -> None:
        result = parse_filename("r01c01f01p01-ch1sk1fk1fl1.tiff")
        assert result is not None
        key, ch = result
        assert key.row == 1
        assert key.col == 1
        assert key.field == 1
        assert ch == 1

    def test_multi_digit_channel(self) -> None:
        result = parse_filename("r15c24f09p01-ch8sk1fk1fl1.tiff")
        assert result is not None
        key, ch = result
        assert key.row == 15
        assert key.col == 24
        assert key.field == 9
        assert ch == 8

    def test_returns_none_for_non_tiff(self) -> None:
        assert parse_filename("Index.idx.xml") is None
        assert parse_filename("readme.txt") is None

    def test_returns_none_for_malformed(self) -> None:
        assert parse_filename("r1c1f1p1-ch1.tiff") is None  # missing padding
        assert parse_filename("random_file.tiff") is None


class TestImageKey:
    def test_well_property(self) -> None:
        key = ImageKey(row=1, col=1, field=1)
        assert key.well == "A01"

    def test_well_property_max(self) -> None:
        key = ImageKey(row=16, col=24, field=9)
        assert key.well == "P24"

    def test_str(self) -> None:
        key = ImageKey(row=3, col=12, field=7)
        assert str(key) == "r03c12f07"

    def test_frozen(self) -> None:
        key = ImageKey(row=1, col=1, field=1)
        with pytest.raises(AttributeError):
            key.row = 2  # type: ignore[misc]

    def test_equality(self) -> None:
        k1 = ImageKey(row=1, col=1, field=1)
        k2 = ImageKey(row=1, col=1, field=1)
        assert k1 == k2

    def test_hashable(self) -> None:
        k1 = ImageKey(row=1, col=1, field=1)
        k2 = ImageKey(row=1, col=1, field=2)
        s = {k1, k2}
        assert len(s) == 2


class TestConstants:
    def test_fluorescence_channels(self) -> None:
        assert FLUORESCENCE_CHANNELS == (1, 2, 3, 4, 5)

    def test_channel_names_keys(self) -> None:
        assert set(CHANNEL_NAMES.keys()) == set(FLUORESCENCE_CHANNELS)

    def test_dino_input_size(self) -> None:
        assert DINO_INPUT_SIZE == 384


class TestDiscoverSites:
    def test_finds_complete_sites(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        assert len(sites) > 0
        # Each site should have all 5 fluorescence channels
        for _key, ch_paths in sites.items():
            assert set(ch_paths.keys()) == set(FLUORESCENCE_CHANNELS)

    def test_keys_have_correct_format(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        for key in sites:
            assert 1 <= key.row <= 16
            assert 1 <= key.col <= 24
            assert key.field >= 1


class TestLoadSingleChannel:
    def test_shape_and_dtype(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        ch_path = sites[key][1]  # channel 1
        arr = load_single_channel(ch_path)
        assert arr.dtype == np.float32
        assert arr.ndim == 2
        assert arr.shape[0] > 0 and arr.shape[1] > 0

    def test_normalized_range(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        ch_path = sites[key][1]
        arr = load_single_channel(ch_path, normalize=True)
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0

    def test_unnormalized(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        ch_path = sites[key][1]
        arr = load_single_channel(ch_path, normalize=False)
        # Should be in uint16 range but as float32
        assert arr.max() <= 65535.0


class TestLoadSite:
    def test_shape(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        arr = load_site(sites[key])
        assert arr.shape[0] == 5  # 5 fluorescence channels
        assert arr.ndim == 3  # (C, H, W)


class TestLoadSiteAsTensor:
    def test_no_resize(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        tensor = load_site_as_tensor(sites[key])
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 5

    def test_resize(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        tensor = load_site_as_tensor(sites[key], resize=224)
        assert tensor.shape == (5, 224, 224)

    def test_resize_custom(self, sample_plate_dir: Path) -> None:
        sites = discover_sites(sample_plate_dir)
        key = next(iter(sites))
        tensor = load_site_as_tensor(sites[key], resize=128)
        assert tensor.shape == (5, 128, 128)


class TestPrepareChannelsForDino:
    def test_output_shape(self) -> None:
        site_tensor = torch.rand(5, 224, 224)
        result = prepare_channels_for_dino(site_tensor)
        assert result.shape == (5, 3, 224, 224)

    def test_channels_replicated(self) -> None:
        site_tensor = torch.rand(5, 32, 32)
        result = prepare_channels_for_dino(site_tensor)
        # Each RGB channel should be the same (grayscale replicated)
        for i in range(5):
            assert torch.allclose(result[i, 0], result[i, 1])
            assert torch.allclose(result[i, 0], result[i, 2])

    def test_with_imagenet_norm(self) -> None:
        site_tensor = torch.ones(5, 32, 32) * 0.5
        result = prepare_channels_for_dino(site_tensor, apply_imagenet_norm=True)
        # After normalization, values should differ from 0.5
        assert not torch.allclose(result[0, 0], torch.ones(32, 32) * 0.5)

    def test_preserves_dtype(self) -> None:
        site_tensor = torch.rand(5, 32, 32, dtype=torch.float32)
        result = prepare_channels_for_dino(site_tensor)
        assert result.dtype == torch.float32
