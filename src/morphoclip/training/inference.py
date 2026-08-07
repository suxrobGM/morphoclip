"""Shared checkpoint loading and model construction for inference/evaluation.

Both ``scripts/training/infer.py`` and ``scripts/training/eval.py`` need
to reconstruct models from a saved checkpoint.  This module provides the
shared plumbing so the scripts stay thin CLI wrappers.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from morphoclip.data.dataset import MorphoCLIPDataset, collate_fn
from morphoclip.data.metadata import MetadataIndex
from morphoclip.models.image_encoder import MorphoCLIPImageEncoder
from morphoclip.models.projection_head import ProjectionHead
from morphoclip.training.config import (
    EMBED_DIM,
    INPUT_CHANNELS,
    PROJ_HIDDEN_DIM,
    TEXT_INPUT_DIM,
    MorphoCLIPTrainingConfig,
    training_config_from_dict,
)
from morphoclip.utils.device import loader_workers, supports_pin_memory


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, MorphoCLIPTrainingConfig]:
    """Load checkpoint and reconstruct config.

    Raises:
        ValueError: If the checkpoint has no embedded config, or its config
            carries a key the current schema does not define.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' key. "
            "Was it saved by the MorphoCLIP trainer?"
        )

    config = training_config_from_dict(ckpt["config"])
    return ckpt, config


def build_models(
    config: MorphoCLIPTrainingConfig,
    device: torch.device,
) -> tuple[MorphoCLIPImageEncoder, ProjectionHead]:
    """Instantiate image encoder and text projection from config."""
    m = config.model
    image_encoder = MorphoCLIPImageEncoder(
        embed_dim=EMBED_DIM,
        output_dim=m.output_dim,
        aggregator=m.aggregator,
        ccf_layers=m.ccf_layers,
        ccf_heads=m.ccf_heads,
        input_channels=INPUT_CHANNELS,
        proj_hidden_dim=PROJ_HIDDEN_DIM,
        proj_dropout=m.proj_dropout,
    ).to(device)

    text_projection = ProjectionHead(
        input_dim=TEXT_INPUT_DIM,
        hidden_dim=PROJ_HIDDEN_DIM,
        output_dim=m.output_dim,
        dropout=m.proj_dropout,
    ).to(device)

    return image_encoder, text_projection


def discover_plates(feature_root: Path) -> list[str]:
    """Return sorted plate directory names that contain ``.pt`` features."""
    return sorted(d.name for d in feature_root.iterdir() if d.is_dir() and any(d.glob("*.pt")))


def build_eval_dataset(
    config: MorphoCLIPTrainingConfig,
    *,
    plates: list[str] | None = None,
    exclude_controls: bool | None = None,
) -> MorphoCLIPDataset:
    """Construct a ``MorphoCLIPDataset`` for eval/inference."""
    ds_cfg = config.dataset
    metadata = MetadataIndex.from_config(Path(ds_cfg.dataset_config_path))
    feature_root = Path(ds_cfg.feature_root)
    if plates is None:
        plates = discover_plates(feature_root)
    return MorphoCLIPDataset(
        feature_dir=feature_root,
        metadata=metadata,
        plates=plates,
        mode="features",
        text_level=ds_cfg.text_level,
        exclude_controls=ds_cfg.exclude_controls if exclude_controls is None else exclude_controls,
        max_sites_per_well=ds_cfg.max_sites_per_well,
    )


def build_eval_dataloader(
    dataset,
    config: MorphoCLIPTrainingConfig,
    device: torch.device,
    *,
    batch_size: int | None = None,
) -> DataLoader:
    """Construct a non-shuffling DataLoader matching trainer conventions.

    Args:
        dataset: Dataset to iterate.
        config: Training config supplying the default eval batch size.
        device: Device the batches feed.
        batch_size: Override for ``config.dataset.eval_batch_size``.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size or config.dataset.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        # build_eval_dataset never preloads, so the dataset reads from disk.
        num_workers=loader_workers(preloaded=False),
        pin_memory=supports_pin_memory(device),
    )


def filter_batch_to_cached(batch: dict, text_cache: dict) -> tuple[dict, int]:
    """Drop samples whose ``broad_sample`` is not in the text cache.

    Returns:
        ``(filtered_batch, n_skipped)``. ``filtered_batch`` is the original
        batch if all samples are cached, else a new dict with filtered
        tensors/lists. ``n_skipped`` is the number of dropped samples.
    """
    id_to_idx = text_cache["id_to_idx"]
    pert_infos = batch["pert_info"]
    valid = [i for i, info in enumerate(pert_infos) if info.broad_sample in id_to_idx]
    if len(valid) == len(pert_infos):
        return batch, 0
    if not valid:
        return {**batch, "pert_info": []}, len(pert_infos)

    idx = torch.tensor(valid, device=batch["features"].device)
    filtered = {
        **batch,
        "features": batch["features"][idx],
        "site_mask": batch["site_mask"][idx],
        "pert_info": [pert_infos[i] for i in valid],
        "plates": [batch["plates"][i] for i in valid],
        "wells": [batch["wells"][i] for i in valid],
    }
    return filtered, len(pert_infos) - len(valid)


def load_models_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[MorphoCLIPImageEncoder, ProjectionHead, dict, MorphoCLIPTrainingConfig]:
    """Load checkpoint, build models, and load weights."""
    ckpt, config = load_checkpoint(checkpoint_path, device)
    image_encoder, text_projection = build_models(config, device)
    image_encoder.load_state_dict(ckpt["image_encoder"])
    text_projection.load_state_dict(ckpt["text_projection"])
    image_encoder.eval()
    text_projection.eval()
    return image_encoder, text_projection, ckpt, config
