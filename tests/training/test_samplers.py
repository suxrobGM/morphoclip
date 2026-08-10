"""Tests for the perturbation-aware batch sampler."""

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from morphoclip.training.samplers import PerturbationBatchSampler, resolve_base_dataset


def _synthetic_keys(
    *, n_groups: int, replicates: int, n_plates: int
) -> tuple[list[str], list[str]]:
    """Build group/plate keys where each group's replicates span all plates."""
    group_keys: list[str] = []
    plate_keys: list[str] = []
    for g in range(n_groups):
        for r in range(replicates):
            group_keys.append(f"BRD-{g:03d}")
            plate_keys.append(f"PLATE{(g + r) % n_plates}")
    return group_keys, plate_keys


def _sampler(
    *,
    n_groups: int = 30,
    replicates: int = 4,
    n_plates: int = 4,
    batch_size: int = 16,
    seed: int = 0,
) -> PerturbationBatchSampler:
    group_keys, plate_keys = _synthetic_keys(
        n_groups=n_groups, replicates=replicates, n_plates=n_plates
    )
    return PerturbationBatchSampler(
        group_keys, plate_keys, batch_size=batch_size, replicates_per_group=2, seed=seed
    )


class TestResolveBaseDataset:
    def test_a_plain_dataset_is_its_own_base(self) -> None:
        dataset = TensorDataset(torch.arange(5))
        assert resolve_base_dataset(dataset) == (dataset, [0, 1, 2, 3, 4])

    def test_nested_subsets_resolve_to_base_indices(self) -> None:
        dataset = TensorDataset(torch.arange(20))
        inner = Subset(dataset, list(range(10, 20)))
        nested = Subset(Subset(inner, [0, 5, 9]), [2, 1])

        base, indices = resolve_base_dataset(nested)
        assert base is dataset
        assert indices == [19, 15]


class TestPerturbationBatchSampler:
    @pytest.mark.parametrize("n_plates", [1, 3])
    def test_an_epoch_covers_every_index_once_in_batches_of_at_most_batch_size(
        self, n_plates: int
    ) -> None:
        sampler = _sampler(n_groups=13, replicates=3, n_plates=n_plates, batch_size=8)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            assert sorted(pos for batch in batches for pos in batch) == list(range(39))
            assert max(len(batch) for batch in batches) <= 8
            assert len(batches) == len(sampler)

    def test_replicates_are_co_batched(self) -> None:
        """Every group present in a batch contributes whole chunks of two wells."""
        group_keys, _ = _synthetic_keys(n_groups=20, replicates=4, n_plates=4)
        sampler = _sampler(n_groups=20, replicates=4, n_plates=4)
        for batch in sampler:
            counts: dict[str, int] = {}
            for pos in batch:
                counts[group_keys[pos]] = counts.get(group_keys[pos], 0) + 1
            assert all(count % 2 == 0 for count in counts.values())

    def test_at_least_two_plates_per_batch(self) -> None:
        _, plate_keys = _synthetic_keys(n_groups=40, replicates=4, n_plates=6)
        sampler = _sampler(n_groups=40, replicates=4, n_plates=6, batch_size=12)
        for batch in sampler:
            assert len({plate_keys[pos] for pos in batch}) >= 2

    def test_singleton_groups_are_not_dropped(self) -> None:
        sampler = PerturbationBatchSampler(
            [f"G{i}" for i in range(7)],
            ["P0", "P1"] * 3 + ["P0"],
            batch_size=3,
            replicates_per_group=2,
        )
        assert sorted(pos for batch in sampler for pos in batch) == list(range(7))

    def test_set_epoch_changes_order(self) -> None:
        sampler = _sampler(seed=7)
        epoch0 = [list(b) for b in sampler]
        sampler.set_epoch(1)
        assert [list(b) for b in sampler] != epoch0

    def test_same_seed_and_epoch_reproduce_the_batches(self) -> None:
        a, b = _sampler(seed=7), _sampler(seed=7)
        a.set_epoch(3)
        b.set_epoch(3)
        assert [list(x) for x in a] == [list(x) for x in b]

    def test_different_seeds_differ(self) -> None:
        assert [list(x) for x in _sampler(seed=1)] != [list(x) for x in _sampler(seed=2)]

    @pytest.mark.parametrize(
        ("group_keys", "plate_keys", "kwargs", "message"),
        [
            (["A", "B"], ["P0"], {"batch_size": 2}, "same length"),
            (["A"], ["P0"], {"batch_size": 0}, "batch_size"),
            (["A"], ["P0"], {"batch_size": 2, "replicates_per_group": 0}, "replicates_per_group"),
        ],
    )
    def test_malformed_construction_raises(
        self, group_keys: list[str], plate_keys: list[str], kwargs: dict, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            PerturbationBatchSampler(group_keys, plate_keys, **kwargs)
