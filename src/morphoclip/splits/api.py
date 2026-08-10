"""Dispatch a split strategy by name into torch ``Subset`` objects.

Strategy implementations live in :mod:`morphoclip.splits.strategies` and the
metadata they read in :mod:`morphoclip.splits.contexts`.
"""

import logging

from torch.utils.data import Subset

from morphoclip.data.dataset import MorphoCLIPDataset, MorphoCLIPSample
from morphoclip.splits.strategies import STRATEGIES, SplitParams

logger = logging.getLogger(__name__)


def _strategy(name: str):
    try:
        return STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown split strategy {name!r}. Known strategies: {known}") from None


def resolve_split_indices(
    dataset: MorphoCLIPDataset,
    strategy: str,
    params: SplitParams,
) -> tuple[list[int], list[int], list[int]]:
    """Return ``(train, validate, test)`` dataset indices for *strategy*.

    Raises:
        ValueError: If *strategy* is not a known name.
    """
    subsets = _strategy(strategy).subsets(dataset, params)
    return subsets["train"], subsets["validate"], subsets["test"]


def create_splits(
    dataset: MorphoCLIPDataset,
    strategy: str = "pert_type",
    params: SplitParams | None = None,
) -> tuple[Subset[MorphoCLIPSample], Subset[MorphoCLIPSample], Subset[MorphoCLIPSample]]:
    """Split *dataset* into train/validate/test subsets.

    Args:
        dataset: The full dataset.
        strategy: A key of :data:`morphoclip.splits.strategies.STRATEGIES`.
        params: Only the ``pert_type`` strategy reads these. The benchmark
            strategies take their partition from reference metadata.

    Returns:
        ``(train, validate, test)`` subsets.
    """
    train_idx, val_idx, test_idx = resolve_split_indices(dataset, strategy, params or SplitParams())
    logger.info(
        "Split %s: train=%d, validate=%d, test=%d",
        strategy,
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def build_split_groups(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
) -> dict[str, list[int]]:
    """Return grouping keys for *strategy*.

    Raises:
        ValueError: If *strategy* is unknown or has no grouping notion.
    """
    groups = _strategy(strategy).groups
    if groups is None:
        raise ValueError(f"Split strategy {strategy!r} does not define groups")
    return groups(dataset)
