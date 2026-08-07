"""Data loading and split creation for MorphoCLIP training."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from morphoclip.data.dataset import MorphoCLIPDataset, collate_fn
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import extract_plate_barcode
from morphoclip.data.splits import create_splits
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import DistributedState
from morphoclip.training.samplers import PerturbationBatchSampler, resolve_base_dataset
from morphoclip.utils.device import loader_workers, supports_pin_memory

EpochSampler = DistributedSampler | PerturbationBatchSampler


def build_sampler_keys(
    subset: Dataset,
    metadata: MetadataIndex,
) -> tuple[list[str], list[str]]:
    """Build per-position perturbation and plate keys for a (possibly nested) subset.

    Uses metadata lookups only, so no feature tensors are loaded.

    Args:
        subset: Training subset, possibly wrapped in nested ``Subset``s.
        metadata: Metadata index for perturbation lookup.

    Returns:
        ``(group_keys, plate_keys)``, both in sampler-position order.

    Raises:
        TypeError: If the underlying dataset is not a ``MorphoCLIPDataset``.
    """
    base, base_indices = resolve_base_dataset(subset)
    if not isinstance(base, MorphoCLIPDataset):
        raise TypeError(
            f"Expected a MorphoCLIPDataset behind the subset, got {type(base).__name__}"
        )

    entries = base.index_entries
    group_keys: list[str] = []
    plate_keys: list[str] = []
    for base_idx in base_indices:
        plate, well, _ = entries[base_idx]
        barcode = extract_plate_barcode(plate)
        group_keys.append(metadata.lookup(barcode, well).broad_sample)
        plate_keys.append(barcode)
    return group_keys, plate_keys


def build_train_data(
    config: MorphoCLIPTrainingConfig,
    device: torch.device,
    *,
    dist_state: DistributedState | None = None,
) -> tuple[DataLoader, DataLoader, int, int, EpochSampler | None]:
    """Build datasets, splits, and data loaders.

    Returns:
        ``(train_loader, val_loader, train_count, val_count, train_sampler)``,
        where ``train_sampler`` is whichever sampler needs ``set_epoch``:
        ``DistributedSampler`` under DDP, ``PerturbationBatchSampler`` when
        ``dataset.batch_sampler`` is ``"perturbation"``, else ``None``.
    """
    ds_cfg = config.dataset
    metadata = MetadataIndex.from_config(Path(ds_cfg.dataset_config_path))
    feature_root = Path(ds_cfg.feature_root)
    plates = sorted(d.name for d in feature_root.iterdir() if d.is_dir() and any(d.glob("*.pt")))

    dataset = MorphoCLIPDataset(
        feature_dir=feature_root,
        metadata=metadata,
        plates=plates,
        mode="features",
        text_level=ds_cfg.text_level,
        exclude_controls=ds_cfg.exclude_controls,
        max_sites_per_well=ds_cfg.max_sites_per_well,
    )
    train_set, val_set, _test_set = create_splits(
        dataset,
        strategy=ds_cfg.split_strategy,
        val_fraction=ds_cfg.val_fraction,
        seed=config.runtime.seed,
    )

    # Preload train+val features into RAM for fast training
    if ds_cfg.preload:
        used_indices = set(train_set.indices + val_set.indices)
        dataset.preload(indices=used_indices)

    dist_sampler: DistributedSampler | None = None
    use_ddp = dist_state is not None and dist_state.world_size > 1
    if use_ddp:
        assert dist_state is not None  # implied by use_ddp
        dist_sampler = DistributedSampler(
            train_set,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
        )

    batch_sampler: PerturbationBatchSampler | None = None
    if ds_cfg.batch_sampler == "perturbation":
        if use_ddp:
            raise ValueError(
                "dataset.batch_sampler='perturbation' does not shard across ranks "
                "and cannot be combined with DDP; use 'random' or run single-GPU."
            )
        group_keys, plate_keys = build_sampler_keys(train_set, metadata)
        batch_sampler = PerturbationBatchSampler(
            group_keys,
            plate_keys,
            batch_size=ds_cfg.batch_size,
            replicates_per_group=ds_cfg.replicates_per_group,
            seed=config.runtime.seed,
        )
    elif ds_cfg.batch_sampler != "random":
        raise ValueError(
            f"Unknown dataset.batch_sampler {ds_cfg.batch_sampler!r} "
            "(expected 'random' or 'perturbation')"
        )

    num_workers = loader_workers(preloaded=ds_cfg.preload)
    pin = supports_pin_memory(device)
    if batch_sampler is not None:
        # DataLoader rejects batch_size/shuffle/sampler alongside batch_sampler.
        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=ds_cfg.batch_size,
            shuffle=(dist_sampler is None),
            sampler=dist_sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin,
        )
    val_loader = DataLoader(
        val_set,
        batch_size=ds_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin,
    )
    train_sampler: EpochSampler | None = batch_sampler or dist_sampler
    return train_loader, val_loader, len(train_set), len(val_set), train_sampler
