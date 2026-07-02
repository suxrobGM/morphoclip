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
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.utils.device import resolve_num_workers, supports_pin_memory


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict, MorphoCLIPTrainingConfig]:
    """Load checkpoint and reconstruct config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has no 'config' key. "
            "Was it saved by the MorphoCLIP trainer?"
        )

    config = MorphoCLIPTrainingConfig()
    config_dict = ckpt["config"]
    for section_name in (
        "dataset",
        "model",
        "optimization",
        "runtime",
        "tensorboard",
        "distributed",
    ):
        section = getattr(config, section_name)
        for key, value in config_dict.get(section_name, {}).items():
            if hasattr(section, key):
                setattr(section, key, value)

    if not isinstance(config.optimization.betas, tuple):
        config.optimization.betas = tuple(config.optimization.betas)

    return ckpt, config


def build_models(
    config: MorphoCLIPTrainingConfig,
    device: torch.device,
) -> tuple[MorphoCLIPImageEncoder, ProjectionHead]:
    """Instantiate image encoder and text projection from config."""
    m = config.model
    image_encoder = MorphoCLIPImageEncoder(
        embed_dim=m.embed_dim,
        output_dim=m.output_dim,
        channel_aggregation=m.channel_aggregation,
        ccf_layers=m.ccf_layers,
        ccf_heads=m.ccf_heads,
        input_channels=m.input_channels,
        proj_hidden_dim=m.proj_hidden_dim,
        proj_dropout=m.proj_dropout,
    ).to(device)

    text_projection = ProjectionHead(
        input_dim=m.text_input_dim,
        hidden_dim=m.proj_hidden_dim,
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
) -> DataLoader:
    """Construct a non-shuffling DataLoader matching trainer conventions."""
    ds_cfg = config.dataset
    return DataLoader(
        dataset,
        batch_size=ds_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=resolve_num_workers(ds_cfg.num_workers),
        pin_memory=ds_cfg.pin_memory and supports_pin_memory(device),
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
