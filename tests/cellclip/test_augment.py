"""Tests for CellCLIP feature-space augmentations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from cellclip.training.augment import FeatureBagAugmenter
from morphoclip.data.perturbation import PerturbationType


class _DummyMetadata:
    def __init__(self, mapping):
        self._mapping = mapping

    def lookup(self, plate: str, well: str):
        return self._mapping[(plate, well)]


class _DummyDataset:
    def __init__(self, entries, features_by_index, metadata):
        self._entries = entries
        self._features_by_index = features_by_index
        self.metadata = metadata

    @property
    def index_entries(self):
        return self._entries

    def __getitem__(self, index: int):
        plate, well, _ = self._entries[index]
        return SimpleNamespace(
            features=self._features_by_index[index].clone(),
            plate=plate,
            well=well,
            pert_info=self.metadata.lookup(plate, well),
        )


def _make_info(broad_sample: str, pert_type: PerturbationType = PerturbationType.COMPOUND):
    return SimpleNamespace(broad_sample=broad_sample, pert_type=pert_type)


def test_feature_bag_augmenter_within_well_interpolates_in_place() -> None:
    dataset = _DummyDataset(
        entries=[("PLATE_A", "A01", [Path("a"), Path("b"), Path("c")])],
        features_by_index={
            0: torch.tensor(
                [
                    [[1.0, 0.0]],
                    [[0.0, 1.0]],
                    [[0.5, 0.5]],
                ]
            ),
        },
        metadata=_DummyMetadata({("PLATE_A", "A01"): _make_info("BRD-1")}),
    )
    augmenter = FeatureBagAugmenter(
        dataset=dataset,
        train_indices=[0],
        plate_contexts={"PLATE_A": SimpleNamespace(cell_type="A549", timepoint=24)},
        within_well_interp_sites=1,
        same_pert_interp_sites=0,
        interp_alpha=0.4,
    )
    augmenter._sample_lambda = lambda **_kwargs: torch.tensor(0.25)

    features = dataset[0].features.unsqueeze(0)
    augmented = augmenter(
        features,
        site_mask=torch.tensor([[True, True, True]]),
        plates=["PLATE_A"],
        wells=["A01"],
        pert_info=[_make_info("BRD-1")],
    )

    assert augmented.shape == features.shape
    assert not torch.allclose(augmented, features)


def test_feature_bag_augmenter_same_pert_partner_prefers_different_plate() -> None:
    dataset = _DummyDataset(
        entries=[
            ("PLATE_A", "A01", [Path("a")]),
            ("PLATE_A", "A02", [Path("b")]),
            ("PLATE_B", "A01", [Path("c")]),
            ("PLATE_C", "A01", [Path("d")]),
        ],
        features_by_index={i: torch.zeros(1, 1, 2) for i in range(4)},
        metadata=_DummyMetadata(
            {
                ("PLATE_A", "A01"): _make_info("BRD-1"),
                ("PLATE_A", "A02"): _make_info("BRD-1"),
                ("PLATE_B", "A01"): _make_info("BRD-1"),
                ("PLATE_C", "A01"): _make_info("BRD-1"),
            }
        ),
    )
    augmenter = FeatureBagAugmenter(
        dataset=dataset,
        train_indices=[0, 1, 2, 3],
        plate_contexts={
            "PLATE_A": SimpleNamespace(cell_type="A549", timepoint=24),
            "PLATE_B": SimpleNamespace(cell_type="A549", timepoint=24),
            "PLATE_C": SimpleNamespace(cell_type="U2OS", timepoint=24),
        },
        within_well_interp_sites=0,
        same_pert_interp_sites=1,
        interp_alpha=0.4,
    )

    partner = augmenter._resolve_partner_index(
        plate="PLATE_A",
        well="A01",
        info=_make_info("BRD-1"),
    )

    assert partner == 2


def test_feature_bag_augmenter_same_perturbation_interpolates_from_dataset_partner() -> None:
    dataset = _DummyDataset(
        entries=[
            ("PLATE_A", "A01", [Path("a"), Path("b")]),
            ("PLATE_B", "A01", [Path("c"), Path("d")]),
        ],
        features_by_index={
            0: torch.zeros(2, 1, 2),
            1: torch.ones(2, 1, 2),
        },
        metadata=_DummyMetadata(
            {
                ("PLATE_A", "A01"): _make_info("BRD-1"),
                ("PLATE_B", "A01"): _make_info("BRD-1"),
            }
        ),
    )
    augmenter = FeatureBagAugmenter(
        dataset=dataset,
        train_indices=[0, 1],
        plate_contexts={
            "PLATE_A": SimpleNamespace(cell_type="A549", timepoint=24),
            "PLATE_B": SimpleNamespace(cell_type="A549", timepoint=24),
        },
        within_well_interp_sites=0,
        same_pert_interp_sites=1,
        interp_alpha=0.4,
    )
    augmenter._sample_lambda = lambda **_kwargs: torch.tensor(0.25)

    features = dataset[0].features.unsqueeze(0)
    augmented = augmenter(
        features,
        site_mask=torch.tensor([[True, True]]),
        plates=["PLATE_A"],
        wells=["A01"],
        pert_info=[_make_info("BRD-1")],
    )

    assert augmented.shape == features.shape
    assert torch.any(augmented > 0.0)
