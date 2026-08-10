"""Batch samplers for MorphoCLIP training.

:class:`PerturbationBatchSampler` puts replicate wells of the same perturbation
in one batch, so CWCL soft positives and the replicate image-image term have
positives to work with, while keeping at least two plates per batch so CWA
removes a real batch effect rather than a single-plate mean.
"""

import logging
import random
from collections.abc import Iterator, Sequence

from torch.utils.data import Dataset, Sampler, Subset

logger = logging.getLogger(__name__)

MIN_PLATES_PER_BATCH = 2


def resolve_base_dataset(dataset: Dataset) -> tuple[Dataset, list[int]]:
    """Unwrap nested ``Subset``s to the base dataset and each position's index.

    Returned together because the indices only mean anything against the
    dataset they were resolved through, and because both halves have to agree
    on what counts as a wrapper.

    Args:
        dataset: A dataset, possibly wrapped in one or more
            :class:`~torch.utils.data.Subset`s.

    Returns:
        ``(base_dataset, base_indices)``, where ``base_indices[i]`` is the
        base-dataset index of sampler position ``i``.
    """
    if isinstance(dataset, Subset):
        base, inner = resolve_base_dataset(dataset.dataset)
        return base, [inner[i] for i in dataset.indices]
    return dataset, list(range(len(dataset)))  # type: ignore[arg-type]


class PerturbationBatchSampler(Sampler[list[int]]):
    """Group replicate wells into batches while keeping plates mixed.

    Each epoch: group positions by perturbation, shuffle within group, split
    into chunks of *replicates_per_group*, then greedily pack the shuffled
    chunks into batches. One fix-up pass swaps chunks between batches so every
    batch spans at least two plates where possible.

    Batches are built eagerly so ``__len__`` is exact; the training loop uses
    it for the step schedule.

    Args:
        group_keys: Perturbation ID (``broad_sample``) per sampler position.
        plate_keys: Plate barcode per sampler position.
        batch_size: Maximum wells per batch.
        replicates_per_group: Target replicate wells per perturbation chunk.
        seed: Base seed; the per-epoch RNG is derived from ``(seed, epoch)``.

    Raises:
        ValueError: If the key lists differ in length or the sizes are
            not positive.
    """

    def __init__(
        self,
        group_keys: Sequence[str],
        plate_keys: Sequence[str],
        *,
        batch_size: int,
        replicates_per_group: int = 2,
        seed: int = 42,
    ) -> None:
        if len(group_keys) != len(plate_keys):
            raise ValueError("group_keys and plate_keys must have the same length")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if replicates_per_group < 1:
            raise ValueError(f"replicates_per_group must be >= 1, got {replicates_per_group}")

        self._plate_keys = list(plate_keys)
        self._batch_size = batch_size
        self._replicates_per_group = replicates_per_group
        self._seed = seed

        self._groups: dict[str, list[int]] = {}
        for position, key in enumerate(group_keys):
            self._groups.setdefault(key, []).append(position)

        self._batches: list[list[int]] = self._build_batches(0)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle for *epoch* (deterministic given the base seed)."""
        self._batches = self._build_batches(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)

    def _build_chunks(self, rng: random.Random) -> list[list[int]]:
        """Shuffle within each perturbation group and split into chunks."""
        chunks: list[list[int]] = []
        k = self._replicates_per_group
        for key in sorted(self._groups):
            positions = list(self._groups[key])
            rng.shuffle(positions)
            chunks.extend(positions[i : i + k] for i in range(0, len(positions), k))
        rng.shuffle(chunks)
        return chunks

    def _pack(self, chunks: list[list[int]]) -> list[list[list[int]]]:
        """Greedily pack chunks into batches, keeping the last partial batch."""
        batches: list[list[list[int]]] = []
        current: list[list[int]] = []
        current_size = 0
        for chunk in chunks:
            if current and current_size + len(chunk) > self._batch_size:
                batches.append(current)
                current, current_size = [], 0
            current.append(chunk)
            current_size += len(chunk)
        if current:
            batches.append(current)
        return batches

    def _distinct_plates(self, batch: list[list[int]]) -> set[str]:
        return {self._plate_keys[pos] for chunk in batch for pos in chunk}

    def _repair_plate_diversity(self, batches: list[list[list[int]]]) -> None:
        """One pass swapping chunks so batches span >= 2 plates where possible."""
        sizes = [sum(len(chunk) for chunk in batch) for batch in batches]
        for i, batch in enumerate(batches):
            plates = self._distinct_plates(batch)
            if len(plates) >= MIN_PLATES_PER_BATCH:
                continue
            if not self._try_swap(batches, sizes, index=i, plates=plates):
                logger.debug(
                    "Batch %d spans a single plate %s; no compatible swap found",
                    i,
                    sorted(plates),
                )

    def _try_swap(
        self,
        batches: list[list[list[int]]],
        sizes: list[int],
        *,
        index: int,
        plates: set[str],
    ) -> bool:
        """Swap one chunk of ``batches[index]`` with a later, plate-diversifying one.

        A one-chunk batch cannot be repaired: swapping its only chunk just
        moves the single-plate batch to a different plate.
        """
        if len(batches[index]) < 2:
            return False
        own = batches[index][0]
        for j in range(index + 1, len(batches)):
            for c, candidate in enumerate(batches[j]):
                if all(self._plate_keys[pos] in plates for pos in candidate):
                    continue
                new_i = sizes[index] - len(own) + len(candidate)
                new_j = sizes[j] - len(candidate) + len(own)
                if new_i > self._batch_size or new_j > self._batch_size:
                    continue
                batches[index][0], batches[j][c] = candidate, own
                sizes[index], sizes[j] = new_i, new_j
                return True
        return False

    def _build_batches(self, epoch: int) -> list[list[int]]:
        rng = random.Random(f"{self._seed}:{epoch}")
        batches = self._pack(self._build_chunks(rng))
        self._repair_plate_diversity(batches)
        return [[pos for chunk in batch for pos in chunk] for batch in batches]
