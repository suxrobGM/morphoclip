"""Tests for the morphoclip.data.dataset module.

Uses real metadata from data/metadata/ and synthesized .pt feature files.
Benchmark split-strategy tests live in tests/benchmark/test_splits.py.
"""

from pathlib import Path

import pytest
import torch

from morphoclip.data.dataset import (
    MorphoCLIPDataset,
    MorphoCLIPSample,
    collate_fn,
)
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationType

BATCH = "2020_11_04_CPJUMP1"
HIDDEN_DIM = 384
NUM_CHANNELS = 5


@pytest.fixture
def metadata_index(metadata_dir: Path) -> MetadataIndex:
    """Build MetadataIndex from real metadata."""
    return MetadataIndex.from_directory(metadata_dir, batch=BATCH)


@pytest.fixture
def fake_features(tmp_path: Path) -> Path:
    """Create synthetic feature .pt files mimicking extracted features.

    Creates features for plate BR00116991 (compound plate):
    - 4 sites across 2 wells (A01 has fields f01,f02; A03 has fields f01,f02)
    """
    plate_dir = tmp_path / "BR00116991"
    plate_dir.mkdir()

    for row, col in [(1, 1), (1, 3)]:  # A01 and A03
        for field in [1, 2]:
            feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
            filename = f"r{row:02d}c{col:02d}f{field:02d}.pt"
            torch.save(feat, plate_dir / filename)

    return tmp_path


class TestMorphoCLIPDataset:
    def test_basic_construction(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        # 2 wells: A01 and A03
        assert len(ds) == 2

    def test_getitem_returns_sample(
        self, fake_features: Path, metadata_index: MetadataIndex
    ) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        sample = ds[0]
        assert isinstance(sample, MorphoCLIPSample)
        # 2 sites per well, 5 channels, 384 dim
        assert sample.features.shape == (2, NUM_CHANNELS, HIDDEN_DIM)
        assert isinstance(sample.text, str)
        assert len(sample.text) > 0

    def test_exclude_controls(
        self, fake_features: Path, metadata_index: MetadataIndex, tmp_path: Path
    ) -> None:
        """A02 is DMSO (negcon). Add features for it and verify exclusion."""
        plate_dir = tmp_path / "BR00116991"
        feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
        torch.save(feat, plate_dir / "r01c02f01.pt")  # A02

        ds_all = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
            exclude_controls=False,
        )
        ds_no_ctrl = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
            exclude_controls=True,
        )
        assert len(ds_all) > len(ds_no_ctrl)

    def test_pert_type_filter(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
            pert_types={PerturbationType.COMPOUND},
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert sample.pert_info.pert_type == PerturbationType.COMPOUND

    def test_metadata_property(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        assert ds.metadata is metadata_index

    def test_text_levels(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        for level in ["name_only", "name_target", "full"]:
            ds = MorphoCLIPDataset(
                feature_dir=fake_features,
                metadata=metadata_index,
                plates=["BR00116991"],
                text_level=level,
            )
            sample = ds[0]
            assert isinstance(sample.text, str)
            assert len(sample.text) > 0

    def test_max_sites_per_well(self, fake_features: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=fake_features,
            metadata=metadata_index,
            plates=["BR00116991"],
            max_sites_per_well=1,
        )
        sample = ds[0]
        assert sample.features.shape[0] == 1

    def test_missing_plate_dir(self, tmp_path: Path, metadata_index: MetadataIndex) -> None:
        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["NONEXISTENT"],
        )
        assert len(ds) == 0


class TestCollateFn:
    def test_pads_variable_sites(
        self, fake_features: Path, metadata_index: MetadataIndex, tmp_path: Path
    ) -> None:
        """Add a 3rd site to one well so sites differ across wells."""
        plate_dir = tmp_path / "BR00116991"
        feat = torch.randn(NUM_CHANNELS, HIDDEN_DIM)
        torch.save(feat, plate_dir / "r01c01f03.pt")  # A01 now has 3 sites

        ds = MorphoCLIPDataset(
            feature_dir=tmp_path,
            metadata=metadata_index,
            plates=["BR00116991"],
        )
        batch = collate_fn([ds[0], ds[1]])
        assert batch["features"].shape[0] == 2  # batch size
        assert batch["features"].shape[1] == 3  # max sites (A01 has 3)
        assert batch["site_mask"].shape == (2, 3)
        # A01 has 3 sites, A03 has 2 — mask should differ
        assert batch["site_mask"][0].sum() == 3
        assert batch["site_mask"][1].sum() == 2
