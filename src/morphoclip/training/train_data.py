"""Data loading and split creation for MorphoCLIP training."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from morphoclip.data.dataset import MorphoCLIPDataset, collate_fn
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.splits import create_splits
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.distributed import DistributedState
from morphoclip.utils.device import resolve_num_workers, supports_pin_memory


def build_train_data(
    config: MorphoCLIPTrainingConfig,
    device: torch.device,
    *,
    dist_state: DistributedState | None = None,
) -> tuple[DataLoader, DataLoader, int, int, DistributedSampler | None]:
    """Build datasets, splits, and data loaders.

    Returns:
        (train_loader, val_loader, train_count, val_count, train_sampler)
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

    if ds_cfg.max_train_wells and len(train_set) > ds_cfg.max_train_wells:
        train_set = torch.utils.data.Subset(train_set, list(range(ds_cfg.max_train_wells)))
    if ds_cfg.max_eval_wells and len(val_set) > ds_cfg.max_eval_wells:
        val_set = torch.utils.data.Subset(val_set, list(range(ds_cfg.max_eval_wells)))

    train_sampler: DistributedSampler | None = None
    use_ddp = dist_state is not None and dist_state.world_size > 1
    if use_ddp:
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=dist_state.world_size,
            rank=dist_state.rank,
            shuffle=True,
        )

    num_workers = resolve_num_workers(ds_cfg.num_workers)
    pin = ds_cfg.pin_memory and supports_pin_memory(device)
    train_loader = DataLoader(
        train_set,
        batch_size=ds_cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    return train_loader, val_loader, len(train_set), len(val_set), train_sampler
