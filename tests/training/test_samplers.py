"""Tests for the perturbation-aware batch sampler."""

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from morphoclip.training.samplers import (
    PerturbationBatchSampler,
    resolve_base_dataset,
)


def _synthetic_keys(
    *,
    n_groups: int,
    replicates: int,
    n_plates: int,
) -> tuple[list[str], list[str]]:
    """Build group/plate keys where each group's replicates span all plates."""
    group_keys: list[str] = []
    plate_keys: list[str] = []
    for g in range(n_groups):
        for r in range(replicates):
            group_keys.append(f"BRD-{g:03d}")
            plate_keys.append(f"PLATE{(g + r) % n_plates}")
    return group_keys, plate_keys


class TestResolveBaseDataset:
    """Tests for nested-Subset unwrapping and index resolution."""

    def test_plain_dataset(self) -> None:
        dataset = TensorDataset(torch.arange(5))
        base, indices = resolve_base_dataset(dataset)
        assert base is dataset
        assert indices == [0, 1, 2, 3, 4]

    def test_single_subset(self) -> None:
        dataset = TensorDataset(torch.arange(10))
        base, indices = resolve_base_dataset(Subset(dataset, [3, 1, 7]))
        assert base is dataset
        assert indices == [3, 1, 7]

    def test_nested_subset(self) -> None:
        dataset = TensorDataset(torch.arange(10))
        outer = Subset(dataset, [2, 4, 6, 8])
        base, indices = resolve_base_dataset(Subset(outer, [3, 0]))
        assert base is dataset
        # Positions 3 and 0 of `outer` are base indices 8 and 2.
        assert indices == [8, 2]

    def test_triple_nesting(self) -> None:
        dataset = TensorDataset(torch.arange(20))
        a = Subset(dataset, list(range(10, 20)))
        b = Subset(a, [0, 5, 9])
        base, indices = resolve_base_dataset(Subset(b, [2, 1]))
        assert base is dataset
        assert indices == [19, 15]

    def test_indices_address_the_returned_base(self) -> None:
        """The two halves must stay consistent: indices address `base`."""
        dataset = TensorDataset(torch.arange(20))
        nested = Subset(Subset(dataset, list(range(10, 20))), [0, 5, 9])
        base, indices = resolve_base_dataset(nested)
        for position, base_idx in enumerate(indices):
            assert base[base_idx][0].item() == nested[position][0].item()


class TestPerturbationBatchSampler:
    """Tests for batch construction, plate mixing, and determinism."""

    def test_every_index_once_per_epoch(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=13, replicates=3, n_plates=3)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=8, replicates_per_group=2
        )
        for epoch in range(3):
            sampler.set_epoch(epoch)
            seen = [pos for batch in sampler for pos in batch]
            assert sorted(seen) == list(range(len(group_keys)))

    def test_batches_respect_batch_size(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=30, replicates=4, n_plates=5)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2
        )
        assert all(len(batch) <= 16 for batch in sampler)

    def test_replicates_are_co_batched(self) -> None:
        """Every non-singleton group contributes chunks of >= 2 wells to a batch."""
        group_keys, plate_keys = _synthetic_keys(n_groups=20, replicates=4, n_plates=4)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2
        )
        for batch in sampler:
            counts: dict[str, int] = {}
            for pos in batch:
                counts[group_keys[pos]] = counts.get(group_keys[pos], 0) + 1
            # Each group present contributes an even chunk count of 2 wells,
            # since every group has exactly 4 replicates and K=2.
            assert all(count % 2 == 0 for count in counts.values())

    def test_at_least_two_plates_per_batch(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=40, replicates=4, n_plates=6)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=12, replicates_per_group=2
        )
        for batch in sampler:
            assert len({plate_keys[pos] for pos in batch}) >= 2

    def test_single_plate_data_does_not_crash(self) -> None:
        """Nothing to repair when the whole dataset is one plate."""
        group_keys, plate_keys = _synthetic_keys(n_groups=10, replicates=2, n_plates=1)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=8, replicates_per_group=2
        )
        seen = [pos for batch in sampler for pos in batch]
        assert sorted(seen) == list(range(len(group_keys)))

    def test_set_epoch_changes_order(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=30, replicates=4, n_plates=4)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2, seed=7
        )
        epoch0 = [list(b) for b in sampler]
        sampler.set_epoch(1)
        epoch1 = [list(b) for b in sampler]
        assert epoch0 != epoch1

    def test_set_epoch_is_reproducible(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=30, replicates=4, n_plates=4)
        a = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2, seed=7
        )
        b = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2, seed=7
        )
        a.set_epoch(3)
        b.set_epoch(3)
        assert [list(x) for x in a] == [list(x) for x in b]

    def test_different_seeds_differ(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=30, replicates=4, n_plates=4)
        a = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2, seed=1
        )
        b = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=16, replicates_per_group=2, seed=2
        )
        assert [list(x) for x in a] != [list(x) for x in b]

    def test_partial_last_batch_kept(self) -> None:
        """drop_last=False semantics: no sample is discarded."""
        group_keys, plate_keys = _synthetic_keys(n_groups=5, replicates=2, n_plates=2)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=4, replicates_per_group=2
        )
        batches = list(sampler)
        assert sum(len(b) for b in batches) == len(group_keys)
        assert len(batches[-1]) <= 4

    def test_singleton_groups_handled(self) -> None:
        group_keys = [f"G{i}" for i in range(7)]
        plate_keys = ["P0", "P1"] * 3 + ["P0"]
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=3, replicates_per_group=2
        )
        seen = [pos for batch in sampler for pos in batch]
        assert sorted(seen) == list(range(7))

    def test_len_matches_iteration(self) -> None:
        group_keys, plate_keys = _synthetic_keys(n_groups=17, replicates=3, n_plates=3)
        sampler = PerturbationBatchSampler(
            group_keys, plate_keys, batch_size=7, replicates_per_group=2
        )
        assert len(sampler) == len(list(sampler))

    def test_mismatched_key_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            PerturbationBatchSampler(["A", "B"], ["P0"], batch_size=2)

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            PerturbationBatchSampler(["A"], ["P0"], batch_size=0)

    def test_invalid_replicates_raise(self) -> None:
        with pytest.raises(ValueError, match="replicates_per_group"):
            PerturbationBatchSampler(["A"], ["P0"], batch_size=2, replicates_per_group=0)
