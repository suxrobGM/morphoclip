"""Shared config resolution for the CellCLIP export and pipeline commands.

Both ``morphoclip cellclip export`` and ``morphoclip cellclip pipeline`` resolve
the same CLI-override / YAML-config / default cascade for the visual-encoder
export. :func:`resolve_export_settings` collapses that shared logic into one
:class:`ExportSettings`. Divergent defaults (batch, device) stay in the command
bodies and are passed in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

from cellclip.benchmark.checkpoint import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CHECKPOINT_REPO,
    resolve_checkpoint,
)
from cellclip.benchmark.export import normalize_timelines, resolve_path


@dataclass(frozen=True)
class ExportSettings:
    """Resolved settings shared by the CellCLIP export and pipeline commands."""

    experiment_metadata: Path
    source_profiles: Path
    features: Path
    output_profiles: Path
    download_dir: Path | None
    repo_id: str
    filename: str
    device: str
    site_batch_size: int
    input_dim: int
    cell_filter: str | None
    timelines: list[str]
    checkpoint_path: str


def resolve_export_settings(
    *,
    project_root: Path,
    benchmark_config: dict,
    export_config: dict,
    context: str,
    default_device: str,
    experiment_metadata_path: str | None,
    source_profiles_root: str | None,
    feature_root: str | None,
    output_profiles_root: str | None,
    cell_filter: str | None,
    timelines: list[str] | None,
    ckpt_path: str | None,
    checkpoint_repo_id: str | None,
    checkpoint_filename: str | None,
    download_dir: str | None,
    device: str | None,
    site_batch_size: int | None,
    input_dim: int | None,
    model_type: str | None,
    loss_type: str | None,
) -> ExportSettings:
    """Resolve the shared export settings from overrides, config, and defaults.

    Args:
        project_root: Root for resolving relative paths.
        benchmark_config: The ``benchmark`` YAML section (timeline/profiles fallbacks).
        export_config: The ``cellclip_export`` YAML section.
        context: ``"exporter"`` or ``"pipeline"`` — used in the ignored-flag warnings.
        default_device: Device fallback when neither the override nor config sets one
            (export uses ``"auto"``; pipeline uses the dataset extraction device).

    Returns:
        The resolved :class:`ExportSettings`. Downloads the checkpoint if needed.
    """
    resolved_timelines = normalize_timelines(
        timelines
        if timelines is not None
        else export_config.get("timelines", benchmark_config.get("timelines"))
    )
    resolved_cell_filter = cell_filter or export_config.get("cell_filter")

    experiment_metadata = resolve_path(
        experiment_metadata_path
        or export_config.get("experiment_metadata_path")
        or "output/benchmark/input/experiment-metadata.tsv",
        project_root,
    )
    source_profiles = resolve_path(
        source_profiles_root
        or export_config.get("source_profiles_root")
        or benchmark_config.get("profiles_dir", "data/profiles"),
        project_root,
    )
    features = resolve_path(
        feature_root or export_config.get("feature_root") or "data/features_cellclip_base",
        project_root,
    )
    output_profiles = resolve_path(
        output_profiles_root
        or export_config.get("output_profiles_root")
        or "data/profiles_cellclip_hf",
        project_root,
    )
    resolved_download_dir = resolve_path(download_dir, project_root) if download_dir else None

    resolved_repo_id = (
        checkpoint_repo_id or export_config.get("checkpoint_repo_id") or DEFAULT_CHECKPOINT_REPO
    )
    resolved_filename = (
        checkpoint_filename
        or export_config.get("checkpoint_filename")
        or DEFAULT_CHECKPOINT_FILENAME
    )
    resolved_device = device or export_config.get("device") or default_device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_site_batch_size = int(site_batch_size or export_config.get("site_batch_size", 16))
    resolved_input_dim = int(input_dim or export_config.get("input_dim", 1536))

    if model_type is not None:
        warnings.warn(
            f"--model-type is ignored by the local CellCLIP {context}; "
            "only the visual encoder is loaded.",
            stacklevel=3,
        )
    if loss_type is not None:
        warnings.warn(
            f"--loss-type is ignored by the local CellCLIP {context}; "
            "it is not needed for image-only export.",
            stacklevel=3,
        )

    checkpoint_path = resolve_checkpoint(
        ckpt_path=ckpt_path,
        checkpoint_repo_id=resolved_repo_id,
        checkpoint_filename=resolved_filename,
        download_dir=resolved_download_dir,
    )

    return ExportSettings(
        experiment_metadata=experiment_metadata,
        source_profiles=source_profiles,
        features=features,
        output_profiles=output_profiles,
        download_dir=resolved_download_dir,
        repo_id=resolved_repo_id,
        filename=resolved_filename,
        device=resolved_device,
        site_batch_size=resolved_site_batch_size,
        input_dim=resolved_input_dim,
        cell_filter=resolved_cell_filter,
        timelines=resolved_timelines,
        checkpoint_path=checkpoint_path,
    )
