"""Tests for morphoclip.data.image_loader."""

from pathlib import Path

import numpy as np
import pytest
import torch

from morphoclip.data.image_loader import (
    FLUORESCENCE_CHANNELS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageKey,
    discover_sites,
    load_single_channel,
    load_site,
    load_site_as_tensor,
    parse_filename,
    prepare_channels_for_dino,
)
from tests.support.images import DEFAULT_SIZE, make_plate_images

PLATE_WELLS = ("A01", "A03")
PLATE_SITES = 2
INCOMPLETE_WELL = "B05"
UINT16_MAX = 65535.0


@pytest.fixture(scope="module")
def sample_plate_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """2 wells x 2 sites, plus brightfield channels and one site missing ch5."""
    image_dir = tmp_path_factory.mktemp("plate") / "Images"
    return make_plate_images(
        image_dir,
        wells=PLATE_WELLS,
        sites=PLATE_SITES,
        incomplete_well=INCOMPLETE_WELL,
    )


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("r01c01f01p01-ch1sk1fk1fl1.tiff", (ImageKey(row=1, col=1, field=1), 1)),
        ("r15c24f09p01-ch8sk1fk1fl1.tiff", (ImageKey(row=15, col=24, field=9), 8)),
        ("Index.idx.xml", None),
        ("readme.txt", None),
        ("r1c1f1p1-ch1.tiff", None),
        ("random_file.tiff", None),
    ],
)
def test_parse_filename_reads_the_site_and_channel_or_rejects_the_name(
    filename: str, expected: tuple[ImageKey, int] | None
) -> None:
    assert parse_filename(filename) == expected


@pytest.mark.parametrize(
    ("key", "well", "site_id"),
    [
        (ImageKey(row=1, col=1, field=1), "A01", "r01c01f01"),
        (ImageKey(row=16, col=24, field=9), "P24", "r16c24f09"),
    ],
)
def test_an_image_key_renders_its_well_and_site_id(key: ImageKey, well: str, site_id: str) -> None:
    assert (key.well, str(key)) == (well, site_id)


def test_discovery_keeps_only_sites_holding_every_fluorescence_channel(
    sample_plate_dir: Path,
) -> None:
    sites = discover_sites(sample_plate_dir)
    complete = {(well, field) for well in PLATE_WELLS for field in range(1, PLATE_SITES + 1)}
    assert {(key.well, key.field) for key in sites} == complete | {(INCOMPLETE_WELL, 2)}
    assert all(set(paths) == set(FLUORESCENCE_CHANNELS) for paths in sites.values())


def test_discovery_drops_brightfield_and_non_image_files(sample_plate_dir: Path) -> None:
    on_disk = {path.name for path in sample_plate_dir.iterdir()}
    assert "Index.idx.xml" in on_disk
    assert any("-ch6" in name for name in on_disk)

    discovered = {
        path.name for paths in discover_sites(sample_plate_dir).values() for path in paths.values()
    }
    assert "Index.idx.xml" not in discovered
    assert not any(f"-ch{channel}" in name for name in discovered for channel in (6, 7, 8))


def test_a_channel_loads_as_float32_normalized_by_the_uint16_maximum(
    sample_plate_dir: Path,
) -> None:
    sites = discover_sites(sample_plate_dir)
    path = sites[next(iter(sites))][1]
    raw = load_single_channel(path, normalize=False)
    normalized = load_single_channel(path)

    assert raw.dtype == np.float32
    assert raw.shape == (DEFAULT_SIZE, DEFAULT_SIZE)
    assert raw.max() > 1.0
    assert np.array_equal(normalized, raw / UINT16_MAX)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_a_site_stacks_its_channels_and_resizes_only_when_asked(sample_plate_dir: Path) -> None:
    sites = discover_sites(sample_plate_dir)
    channel_paths = sites[next(iter(sites))]
    channels = len(FLUORESCENCE_CHANNELS)

    assert load_site(channel_paths).shape == (channels, DEFAULT_SIZE, DEFAULT_SIZE)

    tensor = load_site_as_tensor(channel_paths)
    assert torch.equal(tensor, torch.from_numpy(load_site(channel_paths)))
    assert load_site_as_tensor(channel_paths, resize=128).shape == (channels, 128, 128)


def test_each_channel_becomes_a_replicated_pseudo_rgb_image() -> None:
    site = torch.rand(5, 8, 8)
    result = prepare_channels_for_dino(site)
    assert result.shape == (5, 3, 8, 8)
    assert all(torch.equal(result[:, rgb], site) for rgb in range(3))


def test_imagenet_normalization_uses_the_per_rgb_mean_and_std() -> None:
    result = prepare_channels_for_dino(torch.full((5, 2, 2), 0.5), apply_imagenet_norm=True)
    expected = (0.5 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(result[:, :, 0, 0], expected.expand(5, 3))
