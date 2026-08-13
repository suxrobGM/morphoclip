"""Checkpoint loading, model construction, and well encoding for evaluation.

The `morphoclip eval`, `morphoclip infer` and profile-export commands all
rebuild a model from a checkpoint and then run it over a set of wells. This is
the one place that does either.
"""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from morphoclip.data.dataset import MorphoCLIPDataset, collate_fn
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationInfo
from morphoclip.models.image_encoder import MorphoCLIPImageEncoder
from morphoclip.models.projection_head import ProjectionHead
from morphoclip.training.batch_correction import PlateOffsets, offsets_from_checkpoint
from morphoclip.training.config import (
    EMBED_DIM,
    INPUT_CHANNELS,
    PROJ_HIDDEN_DIM,
    TEXT_INPUT_DIM,
    MorphoCLIPTrainingConfig,
    training_config_from_dict,
)
from morphoclip.utils.device import autocast_context, loader_workers, supports_pin_memory


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
    metadata: MetadataIndex | None = None,
) -> MorphoCLIPDataset:
    """Construct a ``MorphoCLIPDataset`` for eval/inference.

    Args:
        config: Training config supplying the dataset paths and text level.
        plates: Plates to include. Defaults to every plate with cached features.
        exclude_controls: Override ``config.dataset.exclude_controls``.
        metadata: Prebuilt index to reuse. Building one parses the platemaps and
            the four external TSVs, so a caller looping over plates should build
            it once and pass it in.
    """
    ds_cfg = config.dataset
    if metadata is None:
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
    preloaded: bool = False,
) -> DataLoader:
    """Construct a non-shuffling DataLoader matching trainer conventions.

    Args:
        dataset: Dataset to iterate.
        config: Training config supplying the default eval batch size.
        device: Device the batches feed.
        batch_size: Override for ``config.dataset.eval_batch_size``.
        preloaded: Whether ``dataset`` holds its features in memory. Workers
            spawned over a preloaded cache would pickle the whole cache into
            every one of them, so this drops them. Datasets from
            :func:`build_eval_dataset` read from disk and leave it ``False``.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size or config.dataset.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=loader_workers(preloaded=preloaded),
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


@dataclass
class EncodedWells:
    """Everything :func:`encode_wells` collected, on CPU.

    ``text`` is None when no text projection was supplied, which is the
    profile-export case.
    """

    image: torch.Tensor
    text: torch.Tensor | None = None
    plates: list[str] = field(default_factory=list)
    wells: list[str] = field(default_factory=list)
    pert_infos: list[PerturbationInfo] = field(default_factory=list)
    skipped: int = 0

    def require_text(self) -> torch.Tensor:
        """Text embeddings, or an error naming what was not asked for."""
        if self.text is None:
            raise ValueError("encode_wells was called without a text projection")
        return self.text


@torch.no_grad()
def encode_wells(
    image_encoder: nn.Module,
    loader,
    *,
    device: torch.device,
    text_projection: nn.Module | None = None,
    text_cache: dict | None = None,
    amp: bool = False,
    plate_offsets: PlateOffsets | None = None,
) -> EncodedWells:
    """Encode every well in *loader*, optionally alongside its text embedding.

    Args:
        image_encoder: Encoder producing one embedding per well.
        loader: DataLoader over wells.
        device: Device to run on.
        text_projection: Projection head for cached text features. Required to
            populate ``EncodedWells.text``.
        text_cache: Cached text features. When given, wells whose
            ``broad_sample`` is missing from it are dropped and counted in
            ``EncodedWells.skipped``.
        amp: Run the forward pass under autocast. Defaults off, because
            profile export writes its embeddings to disk and fp16 there would
            move every downstream benchmark number.
        plate_offsets: CWA offsets to subtract on-device before the embeddings
            leave the GPU. ``None`` returns raw embeddings, which is what the
            offset-refresh pass itself needs.

    Returns:
        The concatenated embeddings and the per-well metadata that survived.

    Raises:
        ValueError: If *loader* yields no wells at all.
    """
    from morphoclip.training.evaluate import lookup_text_embeddings

    image_embs: list[torch.Tensor] = []
    text_embs: list[torch.Tensor] = []
    result = EncodedWells(image=torch.empty(0))

    for raw_batch in loader:
        batch = raw_batch
        if text_cache is not None:
            batch, n_skipped = filter_batch_to_cached(raw_batch, text_cache)
            result.skipped += n_skipped
        pert_infos: list[PerturbationInfo] = batch["pert_info"]
        if not pert_infos:
            continue

        features = batch["features"].to(device, non_blocking=True)
        site_mask = batch["site_mask"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            image_emb = image_encoder(features, site_mask)
            if plate_offsets is not None:
                image_emb = plate_offsets.apply(image_emb, batch["plates"])
            image_embs.append(image_emb.cpu())
            if text_projection is not None:
                raw_text = lookup_text_embeddings(pert_infos, text_cache or {}, device)
                text_embs.append(text_projection(raw_text).cpu())

        result.plates.extend(batch["plates"])
        result.wells.extend(batch["wells"])
        result.pert_infos.extend(pert_infos)

    if not image_embs:
        raise ValueError("No wells to encode: every batch was empty or uncached")

    result.image = torch.cat(image_embs)
    if text_embs:
        result.text = torch.cat(text_embs)
    return result


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Everything a checkpoint restores, so no consumer has to dig for a piece.

    Attributes:
        image_encoder: Image encoder in eval mode.
        text_projection: Text projection head in eval mode.
        plate_offsets: CWA offsets the run was trained under, ``None`` when CWA
            was off. Part of the model state, not a per-command choice.
        ckpt: The raw checkpoint dict, for epoch, step and logit scale.
        config: The training config the checkpoint was written with.
    """

    image_encoder: MorphoCLIPImageEncoder
    text_projection: ProjectionHead
    plate_offsets: PlateOffsets | None
    ckpt: dict
    config: MorphoCLIPTrainingConfig


def load_models_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedCheckpoint:
    """Load checkpoint, build models, and load weights."""
    ckpt, config = load_checkpoint(checkpoint_path, device)
    image_encoder, text_projection = build_models(config, device)
    image_encoder.load_state_dict(ckpt["image_encoder"])
    text_projection.load_state_dict(ckpt["text_projection"])
    image_encoder.eval()
    text_projection.eval()
    return LoadedCheckpoint(
        image_encoder=image_encoder,
        text_projection=text_projection,
        plate_offsets=offsets_from_checkpoint(ckpt, use_cwa=config.optimization.use_cwa),
        ckpt=ckpt,
        config=config,
    )
