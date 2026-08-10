"""Tests for morphoclip.data.dataset.

Uses the committed CPJUMP1 platemap fixture and a synthesized ``.pt`` feature
tree. Split-strategy tests live in tests/splits/test_splits.py.
"""

from pathlib import Path
from typing import Any

import pytest
import torch

from morphoclip.data.dataset import MorphoCLIPDataset, collate_fn
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationType
from tests.support.constants import HIDDEN_DIM, NUM_CHANNELS
from tests.support.features import make_feature_root

PLATE = "BR00116991"
# A02 is the DMSO control well on this plate; A01 and A03 are compounds.
WELLS = ("A01", "A02", "A03")
SITES = 2


@pytest.fixture
def feature_root(tmp_path: Path) -> Path:
    return make_feature_root(tmp_path / "features", {PLATE: WELLS}, sites=SITES, dim=HIDDEN_DIM)


def _dataset(feature_root: Path, metadata: MetadataIndex, **kwargs: Any) -> MorphoCLIPDataset:
    return MorphoCLIPDataset(feature_dir=feature_root, metadata=metadata, plates=[PLATE], **kwargs)


def _all_features(ds: MorphoCLIPDataset) -> list[torch.Tensor]:
    return [ds[i].features for i in range(len(ds))]


def test_a_sample_stacks_the_wells_sites_and_carries_its_perturbation_text(
    feature_root: Path, metadata_index: MetadataIndex
) -> None:
    ds = _dataset(feature_root, metadata_index)
    sample = ds[0]

    assert len(ds) == len(WELLS)
    assert (sample.plate, sample.well) == (PLATE, "A01")
    assert sample.features.shape == (SITES, NUM_CHANNELS, HIDDEN_DIM)
    assert sample.pert_info.pert_type == PerturbationType.COMPOUND
    assert sample.text.startswith("Chemical perturbation: gabapentin-enacarbil. Target: CACNB4.")


def test_the_text_level_reaches_the_generated_prompt(
    feature_root: Path, metadata_index: MetadataIndex
) -> None:
    ds = _dataset(feature_root, metadata_index, text_level="name_only")
    assert ds[0].text == "Chemical perturbation: gabapentin-enacarbil."


@pytest.mark.parametrize(
    ("options", "expected_wells"),
    [
        ({}, ["A01", "A02", "A03"]),
        ({"exclude_controls": True}, ["A01", "A03"]),
        ({"pert_types": {PerturbationType.COMPOUND}}, ["A01", "A03"]),
        ({"pert_types": {PerturbationType.CRISPR}}, []),
    ],
)
def test_the_index_holds_only_the_wells_the_filters_admit(
    feature_root: Path,
    metadata_index: MetadataIndex,
    options: dict[str, Any],
    expected_wells: list[str],
) -> None:
    ds = _dataset(feature_root, metadata_index, **options)
    assert [well for _plate, well, _paths in ds.index_entries] == expected_wells


def test_max_sites_per_well_caps_the_stacked_sites(
    feature_root: Path, metadata_index: MetadataIndex
) -> None:
    ds = _dataset(feature_root, metadata_index, max_sites_per_well=1)
    assert ds[0].features.shape[0] == 1


def test_a_plate_with_no_feature_directory_contributes_no_wells(
    tmp_path: Path, metadata_index: MetadataIndex
) -> None:
    ds = MorphoCLIPDataset(feature_dir=tmp_path, metadata=metadata_index, plates=["NONEXISTENT"])
    assert len(ds) == 0


class TestPreload:
    """Preloading must change only where tensors come from, never what they are."""

    def test_a_partial_preload_serves_its_own_wells_from_memory_and_the_rest_from_disk(
        self, feature_root: Path, metadata_index: MetadataIndex
    ) -> None:
        """Training preloads train+val only, so test-split wells miss the cache."""
        expected = _all_features(_dataset(feature_root, metadata_index))

        ds = _dataset(feature_root, metadata_index)
        ds.preload(indices={0})
        for _plate, _well, site_paths in ds.index_entries[:1]:
            for path in site_paths:
                path.unlink()

        assert all(torch.equal(a, b) for a, b in zip(_all_features(ds), expected, strict=True))

    def test_preloading_everything_twice_leaves_the_samples_unchanged(
        self, feature_root: Path, metadata_index: MetadataIndex
    ) -> None:
        expected = _all_features(_dataset(feature_root, metadata_index))

        ds = _dataset(feature_root, metadata_index)
        ds.preload()
        ds.preload()

        assert all(torch.equal(a, b) for a, b in zip(_all_features(ds), expected, strict=True))


def test_collate_pads_to_the_widest_well_and_masks_the_padding(
    tmp_path: Path, metadata_index: MetadataIndex
) -> None:
    root = make_feature_root(
        tmp_path / "ragged", {PLATE: ["A01", "A03"]}, sites={"A01": 3, "A03": 2}, dim=HIDDEN_DIM
    )
    ds = _dataset(root, metadata_index)

    batch = collate_fn([ds[0], ds[1]])

    assert batch["features"].shape == (2, 3, NUM_CHANNELS, HIDDEN_DIM)
    assert batch["site_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert torch.equal(batch["features"][1, 2], torch.zeros(NUM_CHANNELS, HIDDEN_DIM))
    assert batch["wells"] == ["A01", "A03"]
