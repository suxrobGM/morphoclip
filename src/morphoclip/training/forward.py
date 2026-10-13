"""The well forward pass shared by training, evaluation and inference."""

from typing import overload

import torch
from torch import nn

from morphoclip.data.perturbation import PerturbationInfo
from morphoclip.training.batch_correction import PlateOffsets


def lookup_text_embeddings(
    pert_infos: list[PerturbationInfo],
    text_cache: dict,
    device: torch.device,
) -> torch.Tensor:
    """Look up cached 768-d BERT features for a batch of perturbations.

    Args:
        pert_infos: PerturbationInfo for each sample in the batch.
        text_cache: Dict from ``load_cached_text_features`` with
            ``embeddings`` and ``id_to_idx``.
        device: Target device.

    Returns:
        Raw BERT features ``(B, 768)`` on *device*.
    """
    id_to_idx = text_cache["id_to_idx"]
    embeddings = text_cache["embeddings"]
    indices = [id_to_idx[info.broad_sample] for info in pert_infos]
    return embeddings[indices].to(device, non_blocking=True)


@overload
def encode_batch(
    batch: dict,
    image_encoder: nn.Module,
    text_projection: nn.Module,
    text_cache: dict,
    *,
    device: torch.device,
    plate_offsets: PlateOffsets | None,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def encode_batch(
    batch: dict,
    image_encoder: nn.Module,
    text_projection: None,
    text_cache: dict | None,
    *,
    device: torch.device,
    plate_offsets: PlateOffsets | None,
) -> tuple[torch.Tensor, None]: ...


@overload
def encode_batch(
    batch: dict,
    image_encoder: nn.Module,
    text_projection: nn.Module | None,
    text_cache: dict | None,
    *,
    device: torch.device,
    plate_offsets: PlateOffsets | None,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...


def encode_batch(
    batch: dict,
    image_encoder: nn.Module,
    text_projection: nn.Module | None,
    text_cache: dict | None,
    *,
    device: torch.device,
    plate_offsets: PlateOffsets | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode one collated batch of wells, applying CWA to the image side.

    Runs outside any autocast context; callers choose the precision.

    Args:
        batch: Collated batch with ``features``, ``site_mask``, ``pert_info``
            and ``plates``.
        image_encoder: Encoder producing one embedding per well.
        text_projection: Projection head over cached BERT features, or
            ``None`` to skip the text side.
        text_cache: Cached text features. Required with *text_projection*.
        device: Device to run on.
        plate_offsets: CWA offsets to subtract, or ``None`` for raw embeddings.

    Returns:
        ``(image_emb, text_emb)``. ``text_emb`` is ``None`` without a
        *text_projection*.

    Raises:
        ValueError: If *text_projection* is given without *text_cache*.
    """
    features = batch["features"].to(device, non_blocking=True)
    site_mask = batch["site_mask"].to(device, non_blocking=True)
    image_emb = image_encoder(features, site_mask)

    text_emb: torch.Tensor | None = None
    if text_projection is not None:
        if text_cache is None:
            raise ValueError("text_cache is required when text_projection is given")
        text_emb = text_projection(lookup_text_embeddings(batch["pert_info"], text_cache, device))

    if plate_offsets is not None:
        image_emb = plate_offsets.apply(image_emb, batch["plates"])
    return image_emb, text_emb
