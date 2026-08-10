"""Export MorphoCLIP well embeddings into benchmark-layout profile CSVs.

Writes per-plate CSVs in the layout ``benchmark.data.ProfileLoader`` reads
(``<out_root>/<batch>/<plate>/<plate>_normalized_feature_select_negcon_batch.csv.gz``),
so the CPJUMP1 benchmark runs against MorphoCLIP embeddings the same way it runs
against CellCLIP or CellProfiler profiles.
"""

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from benchmark.data import get_metadata_columns
from benchmark.export_utils import feature_columns, negcon_center_profiles, output_profile_path
from morphoclip.training.config import MorphoCLIPTrainingConfig
from morphoclip.training.inference import (
    build_eval_dataloader,
    build_eval_dataset,
    encode_wells,
    load_models_from_checkpoint,
)
from morphoclip.utils.device import resolve_device

logger = logging.getLogger(__name__)

type ExportModels = tuple[nn.Module, MorphoCLIPTrainingConfig, torch.device]


class PlateNotInReferenceError(ValueError):
    """Raised when a plate has no rows in the CPJUMP1 reference metadata.

    Expected for plates outside the benchmark-eligible subset (non-standard
    seeding density, antibiotics present, compound plates on the Cas9 line), so
    callers usually skip rather than fail.
    """


def load_export_models(checkpoint: Path, device: str = "auto") -> ExportModels:
    """Load the image encoder and config from a checkpoint, for reuse across plates.

    Args:
        checkpoint: Path to a trained MorphoCLIP checkpoint.
        device: Device string (``"auto"``, ``"cuda"``, ``"cpu"``, ...).

    Returns:
        Tuple of ``(image_encoder, config, device)`` ready for encoding.
    """
    torch_device = resolve_device(device)
    image_encoder, _text_projection, _ckpt, config = load_models_from_checkpoint(
        checkpoint, torch_device
    )
    image_encoder.eval()
    return image_encoder, config, torch_device


@lru_cache(maxsize=4)
def _read_reference_csv(csv_path: Path) -> pd.DataFrame:
    """Read the reference metadata CSV once, shared across plates."""
    return pd.read_csv(csv_path, low_memory=False)


def load_reference_metadata(csv_path: Path, plate: str) -> pd.DataFrame:
    """Load and filter first-party CPJUMP1 reference metadata to one plate.

    Args:
        csv_path: Path to ``cpjump1_metadata.csv``.
        plate: Plate barcode to filter to (matches ``Metadata_Plate``).

    Returns:
        DataFrame with one row per well for ``plate``.

    Raises:
        ValueError: If the CSV lacks ``Metadata_Plate``/``Metadata_Well`` or a
            well appears more than once.
        PlateNotInReferenceError: If no rows match ``plate``.
    """
    df = _read_reference_csv(csv_path)
    for required_col in ("Metadata_Plate", "Metadata_Well"):
        if required_col not in df.columns:
            raise ValueError(f"Reference metadata {csv_path} is missing column {required_col!r}")

    plate_df = df[df["Metadata_Plate"].astype(str) == str(plate)].reset_index(drop=True)
    if plate_df.empty:
        raise PlateNotInReferenceError(
            f"No reference metadata rows found for plate {plate!r} in {csv_path}"
        )

    duplicated = plate_df["Metadata_Well"].duplicated(keep=False)
    if duplicated.any():
        dup_wells = sorted(plate_df.loc[duplicated, "Metadata_Well"].astype(str).unique())
        raise ValueError(
            f"Reference metadata has duplicate Metadata_Well entries for plate {plate!r}: "
            f"{dup_wells}"
        )
    return plate_df


def build_plate_profile(
    embeddings: np.ndarray | torch.Tensor,
    wells: list[str],
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble a benchmark-layout profile DataFrame for one plate.

    Aligns embeddings with metadata rows by ``Metadata_Well``. Wells present on
    only one side are warned about and left out of the output.

    Args:
        embeddings: ``(N, D)`` well embeddings, in the same order as ``wells``.
        wells: Well position strings, one per row of ``embeddings``.
        metadata_df: Reference metadata for the plate (``Metadata_*`` columns),
            as returned by :func:`load_reference_metadata`.

    Returns:
        DataFrame with all ``Metadata_*`` columns from ``metadata_df`` plus
        ``feature_0000``..``feature_{D-1:04d}`` columns, one row per aligned well.

    Raises:
        ValueError: If ``wells`` and ``embeddings`` lengths disagree, ``wells``
            contains duplicates, or no wells align with ``metadata_df``.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings (N, D), got shape {embeddings.shape}")
    if len(wells) != embeddings.shape[0]:
        raise ValueError(
            f"wells length ({len(wells)}) does not match embeddings rows ({embeddings.shape[0]})"
        )

    seen: set[str] = set()
    duplicate_wells: list[str] = []
    for well in wells:
        if well in seen:
            duplicate_wells.append(well)
        seen.add(well)
    if duplicate_wells:
        raise ValueError(f"Duplicate wells in embeddings: {sorted(set(duplicate_wells))}")

    embedding_by_well = dict(zip(wells, embeddings, strict=True))
    metadata_wells = metadata_df["Metadata_Well"].astype(str)

    missing_embeddings = sorted(set(metadata_wells) - set(embedding_by_well))
    if missing_embeddings:
        logger.warning(
            "%d metadata wells have no encoded embedding and will be absent "
            "from the output profile: %s",
            len(missing_embeddings),
            missing_embeddings[:10],
        )

    missing_metadata = sorted(set(embedding_by_well) - set(metadata_wells))
    if missing_metadata:
        logger.warning(
            "Dropping %d encoded wells with no reference metadata: %s",
            len(missing_metadata),
            missing_metadata[:10],
        )

    keep_mask = metadata_wells.isin(embedding_by_well)
    kept_rows = metadata_df.loc[keep_mask].reset_index(drop=True)
    if kept_rows.empty:
        raise ValueError("No wells align between embeddings and reference metadata")

    metadata_cols = get_metadata_columns(metadata_df)
    feature_matrix = np.vstack(
        [embedding_by_well[str(well)] for well in kept_rows["Metadata_Well"]]
    )
    features_df = pd.DataFrame(
        feature_matrix.astype(np.float32),
        columns=feature_columns(feature_matrix.shape[1]),
    )
    metadata_out = kept_rows[metadata_cols].reset_index(drop=True)
    return pd.concat([metadata_out, features_df], axis=1)


def _encode_plate_wells(
    models: ExportModels,
    plate: str,
    *,
    batch_size: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Encode every well cached for ``plate`` through the image encoder.

    Runs in fp32: these embeddings are written to disk and every benchmark
    number downstream is computed from them.
    """
    image_encoder, config, device = models

    dataset = build_eval_dataset(config, plates=[plate], exclude_controls=False)
    if len(dataset) == 0:
        raise ValueError(
            f"No cached features found for plate {plate!r} under {config.dataset.feature_root}"
        )
    loader = build_eval_dataloader(dataset, config, device, batch_size=batch_size)

    encoded = encode_wells(image_encoder, loader, device=device)
    return encoded.image.numpy(), encoded.wells


def export_plate_profiles(
    models: ExportModels,
    plate: str,
    *,
    metadata_csv: Path,
    output_root: Path,
    batch: str = "2020_11_04_CPJUMP1",
    negcon_center: bool = True,
    batch_size: int | None = None,
) -> Path:
    """Export a benchmark-layout profile CSV for one plate.

    Encodes every cached well including controls, aligns them with reference
    metadata, optionally centers against plate negative controls, and writes to
    the path :func:`benchmark.export_utils.output_profile_path` expects.

    Args:
        models: ``(image_encoder, config, device)`` from
            :func:`load_export_models`, loaded once and reused across plates.
        plate: Plate barcode to export.
        metadata_csv: Path to the CPJUMP1 reference metadata CSV.
        output_root: Root directory for exported profiles.
        batch: Batch name used in the output path.
        negcon_center: If ``True``, center features against plate negcon wells.
        batch_size: Override the checkpoint's eval batch size.

    Returns:
        Path to the written profile CSV.
    """
    embeddings, wells = _encode_plate_wells(models, plate, batch_size=batch_size)
    metadata_df = load_reference_metadata(metadata_csv, plate)
    profile_df = build_plate_profile(embeddings, wells, metadata_df)

    logger.info(
        "Plate %s: %d reference wells, %d encoded wells, %d wells in output profile",
        plate,
        len(metadata_df),
        len(wells),
        len(profile_df),
    )

    if negcon_center:
        has_negcon = (
            "Metadata_control_type" in profile_df.columns
            and (profile_df["Metadata_control_type"] == "negcon").any()
        )
        if not has_negcon:
            logger.warning(
                "Plate %s has zero negcon rows with features; falling back to all-well centering",
                plate,
            )
        profile_df = negcon_center_profiles(profile_df)

    output_path = output_profile_path(output_root, batch, plate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(output_path, index=False, compression="gzip")
    return output_path
