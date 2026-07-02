"""Benchmark export helpers for local CellCLIP profile generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from benchmark.data import get_timepoint_label
from morphoclip.data.image_loader import FEATURE_PATTERN
from morphoclip.data.perturbation import well_from_row_col

TIMELINE_CHOICES = ("short", "long")


def resolve_path(path_str: str | Path, project_root: Path) -> Path:
    """Resolve project-relative paths while preserving absolute paths."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


def load_yaml_section(path: Path, section: str) -> dict:
    """Load a named YAML section if present."""
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    data = raw.get(section, {})
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping: {path}")
    return data


def normalize_timelines(value) -> list[str]:
    """Normalize timeline selection to a deduplicated list."""
    if value is None:
        return list(TIMELINE_CHOICES)

    items = [value] if isinstance(value, str) else list(value)
    normalized: list[str] = []
    for item in items:
        label = str(item).strip().lower()
        if label not in TIMELINE_CHOICES:
            raise ValueError(
                f"Invalid timeline {item!r}; expected one of: {', '.join(TIMELINE_CHOICES)}"
            )
        if label not in normalized:
            normalized.append(label)
    return normalized or list(TIMELINE_CHOICES)


def get_profile_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Return metadata columns preserved in exported profile CSVs."""
    return [c for c in df.columns if c.startswith("Metadata_") or c.startswith("Meta_")]


def select_target_plates(
    experiment_metadata_path: Path,
    batch: str,
    cell_filter: str | None,
    timelines: list[str],
) -> list[str]:
    """Select plates from benchmark experiment metadata."""
    experiment_df = (
        pd.read_csv(experiment_metadata_path, sep="\t")
        .query(f"Batch=='{batch}'")
        .query("Density==100")
        .query('Antibiotics=="absent"')
    )
    experiment_df = experiment_df.drop(
        experiment_df[
            (experiment_df.Perturbation == "compound") & (experiment_df.Cell_line == "Cas9")
        ].index
    )
    experiment_df["timeline"] = experiment_df.apply(
        lambda row: get_timepoint_label(row["Perturbation"], row["Time"]),
        axis=1,
    )
    experiment_df = experiment_df[experiment_df["timeline"].isin(timelines)].reset_index(drop=True)

    if cell_filter:
        experiment_df = experiment_df.query("Cell_type==@cell_filter").reset_index(drop=True)

    if experiment_df.empty:
        raise ValueError(
            "No experiments matched the requested export slice: "
            f"batch={batch}, cell_filter={cell_filter!r}, timelines={timelines}"
        )

    return experiment_df["Assay_Plate_Barcode"].drop_duplicates().tolist()


def resolve_feature_dir(feature_root: Path, plate: str) -> Path:
    """Resolve feature directory for a plate barcode."""
    barcode_dir = feature_root / plate
    if barcode_dir.exists():
        return barcode_dir
    raise FileNotFoundError(f"Feature directory not found for plate {plate}: {barcode_dir}")


def load_source_profile(source_profiles_root: Path, batch: str, plate: str) -> pd.DataFrame:
    """Load the source benchmark profile CSV for a plate."""
    profile_path = (
        source_profiles_root
        / batch
        / plate
        / f"{plate}_normalized_feature_select_negcon_batch.csv.gz"
    )
    if not profile_path.exists():
        raise FileNotFoundError(f"Source profile not found for plate {plate}: {profile_path}")

    df = pd.read_csv(profile_path, low_memory=False)
    if "Metadata_Well" not in df.columns:
        raise ValueError(f"Source profile missing Metadata_Well: {profile_path}")
    if df["Metadata_Well"].duplicated().any():
        raise ValueError(
            f"Source profile has duplicate wells; exporter expects one row per well: {profile_path}"
        )
    return df


def build_well_to_site_paths(feature_dir: Path) -> dict[str, list[Path]]:
    """Group site feature files by well."""
    grouped: dict[str, list[Path]] = {}
    for pt_path in sorted(feature_dir.glob("*.pt")):
        match = FEATURE_PATTERN.fullmatch(pt_path.name)
        if match is None:
            continue
        well = well_from_row_col(int(match["row"]), int(match["col"]))
        grouped.setdefault(well, []).append(pt_path)
    return grouped


def load_well_sites(site_paths: list[Path]) -> torch.Tensor:
    """Load all site tensors for a well into shape ``(num_sites, 5, D)``."""
    site_tensors = [torch.load(path, map_location="cpu", weights_only=True) for path in site_paths]
    sites = torch.stack(site_tensors, dim=0).float()
    if sites.ndim != 3:
        raise ValueError(
            f"Expected stacked site tensor shape (num_sites, 5, D), got {tuple(sites.shape)}"
        )
    return sites


@torch.no_grad()
def encode_well(
    model: torch.nn.Module,
    sites: torch.Tensor,
    device: str,
    site_batch_size: int,
) -> np.ndarray:
    """Encode one well using the trainer-faithful MIL pooling path."""
    del site_batch_size
    pooled_sites = model.encode_mil(sites.unsqueeze(0).to(device))
    well_embedding = model.encode_image(pooled_sites).squeeze(0).detach().cpu().float()
    return well_embedding.numpy()


def feature_columns(width: int) -> list[str]:
    """Build exported feature column names."""
    return [f"feature_{i:04d}" for i in range(width)]


def negcon_center_profiles(
    profiles: pd.DataFrame,
    *,
    control_col: str = "Metadata_control_type",
) -> pd.DataFrame:
    """Center exported features against plate-level negative controls.

    The benchmark consumes files named ``normalized_feature_select_negcon_batch``.
    Raw CellCLIP well embeddings have a strong shared offset across wells, so we
    remove the negative-control reference mean before saving.
    """
    feature_cols = [col for col in profiles.columns if not col.startswith("Metadata")]
    if not feature_cols:
        return profiles

    negcon_mask = profiles[control_col].eq("negcon") if control_col in profiles.columns else None
    if negcon_mask is not None and bool(negcon_mask.any()):
        reference = profiles.loc[negcon_mask, feature_cols]
    else:
        reference = profiles[feature_cols]

    centered = profiles.copy()
    reference_mean = reference.to_numpy(dtype=np.float32).mean(axis=0)
    centered[feature_cols] = profiles[feature_cols].to_numpy(dtype=np.float32) - reference_mean
    return centered


def output_profile_path(output_profiles_root: Path, batch: str, plate: str) -> Path:
    """Return the benchmark-compatible exported profile path for a plate."""
    return (
        output_profiles_root
        / batch
        / plate
        / f"{plate}_normalized_feature_select_negcon_batch.csv.gz"
    )


def export_plate(
    *,
    model: torch.nn.Module,
    device: str,
    source_profiles_root: Path,
    feature_root: Path,
    output_profiles_root: Path,
    batch: str,
    plate: str,
    site_batch_size: int,
) -> Path:
    """Export one benchmark-compatible plate profile CSV."""
    source_df = load_source_profile(source_profiles_root, batch, plate)
    metadata_cols = get_profile_metadata_columns(source_df)
    metadata_df = source_df[metadata_cols].copy().reset_index(drop=True)

    feature_dir = resolve_feature_dir(feature_root, plate)
    well_to_paths = build_well_to_site_paths(feature_dir)

    exported_features: list[np.ndarray] = []
    missing_wells: list[str] = []
    embedding_width: int | None = None

    for well in metadata_df["Metadata_Well"].astype(str):
        site_paths = well_to_paths.get(well)
        if not site_paths:
            missing_wells.append(well)
            continue

        sites = load_well_sites(site_paths)
        encoded = encode_well(model, sites, device, site_batch_size)
        if embedding_width is None:
            embedding_width = int(encoded.shape[0])
        exported_features.append(encoded)

    if missing_wells:
        preview = ", ".join(missing_wells[:10])
        raise ValueError(
            f"Missing CellCLIP input features for {len(missing_wells)} wells "
            f"in plate {plate}: {preview}"
        )

    if embedding_width is None:
        raise ValueError(f"No wells were exported for plate {plate}")

    features_df = pd.DataFrame(
        np.vstack(exported_features),
        columns=feature_columns(embedding_width),
    )
    output_df = pd.concat([metadata_df, features_df], axis=1)
    output_df = negcon_center_profiles(output_df)

    output_path = output_profile_path(output_profiles_root, batch, plate)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, compression="gzip")
    return output_path
