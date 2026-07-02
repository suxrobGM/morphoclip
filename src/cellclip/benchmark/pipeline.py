"""Plate-at-a-time CellCLIP export pipeline with disk-bounded cleanup."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cellclip.benchmark.export import (
    export_plate,
    load_yaml_section,
    output_profile_path,
    resolve_path,
)
from morphoclip.data.feature_extractor import (
    extract_plate_features_with_model,
    verify_plate_features,
)
from morphoclip.data.perturbation import extract_plate_barcode

STATUS_STYLES = {
    "skipped_existing_export": "yellow",
    "skipped_missing_images": "yellow",
    "reused_existing_features": "cyan",
    "exported": "green",
    "failed": "red",
}


@dataclass(slots=True)
class PlatePaths:
    """Resolved local paths for one plate barcode."""

    barcode: str
    image_dir: Path
    feature_dir: Path
    tensor_dir: Path
    output_path: Path


@dataclass(slots=True)
class PlateResult:
    """Summary of one plate pipeline run."""

    barcode: str
    status: str
    message: str
    output_path: Path | None = None
    features_reused: bool = False
    features_deleted: int = 0
    tensors_deleted: int = 0


def load_dataset_config(path: Path) -> dict[str, Any]:
    """Load the dataset config section used by the extractor."""
    data = load_yaml_section(path, "cpjump")
    if not data:
        raise ValueError(f"Dataset config missing 'cpjump' section: {path}")
    return data


def resolve_dataset_path(path_value: str | Path, project_root: Path) -> Path:
    """Resolve project-relative dataset paths."""
    return resolve_path(path_value, project_root)


def discover_downloaded_plate_dirs(compressed_root: Path, batch: str) -> dict[str, Path]:
    """Map downloaded plate barcodes to compressed image directories."""
    candidates: list[Path] = []
    batch_root = compressed_root / batch
    if batch_root.exists():
        candidates.extend(path for path in sorted(batch_root.iterdir()) if path.is_dir())
    elif compressed_root.exists():
        candidates.extend(path for path in sorted(compressed_root.iterdir()) if path.is_dir())

    mapping: dict[str, Path] = {}
    for plate_dir in candidates:
        image_dir = plate_dir / "Images"
        if not image_dir.exists():
            continue

        barcode = extract_plate_barcode(plate_dir.name)
        existing = mapping.get(barcode)
        if existing is not None and existing != image_dir:
            raise ValueError(
                f"Multiple compressed-image directories found for plate {barcode}: "
                f"{existing} and {image_dir}"
            )
        mapping[barcode] = image_dir
    return mapping


def resolve_plate_paths(
    *,
    barcodes: list[str],
    compressed_root: Path,
    features_root: Path,
    tensors_root: Path,
    output_profiles_root: Path,
    batch: str,
) -> tuple[list[PlatePaths], list[str]]:
    """Resolve local image, feature, tensor, and output paths for target barcodes."""
    downloaded = discover_downloaded_plate_dirs(compressed_root, batch)

    resolved: list[PlatePaths] = []
    missing: list[str] = []
    for barcode in barcodes:
        image_dir = downloaded.get(barcode)
        if image_dir is None:
            missing.append(barcode)
            continue

        resolved.append(
            PlatePaths(
                barcode=barcode,
                image_dir=image_dir,
                feature_dir=features_root / barcode,
                tensor_dir=tensors_root / barcode,
                output_path=output_profile_path(output_profiles_root, batch, barcode),
            )
        )

    return resolved, missing


def clear_pt_files(directory: Path) -> int:
    """Delete ``.pt`` files from a directory and return the count removed."""
    if not directory.exists():
        return 0

    removed = 0
    for path in directory.glob("*.pt"):
        path.unlink()
        removed += 1
    return removed


def prune_empty_directory(directory: Path) -> None:
    """Remove a directory if it exists and is empty."""
    if directory.exists() and not any(directory.iterdir()):
        directory.rmdir()


def feature_cache_state(feature_dir: Path, image_dir: Path) -> tuple[bool, int, int]:
    """Return ``(is_complete, extracted_count, expected_count)`` for a feature cache."""
    extracted, expected, missing = verify_plate_features(feature_dir, image_dir)
    return extracted == expected and not missing, extracted, expected


def cached_feature_width(feature_dir: Path) -> int | None:
    """Return the cached feature width from one saved site tensor, if available."""
    if not feature_dir.exists():
        return None

    sample_paths = sorted(feature_dir.glob("*.pt"))
    if not sample_paths:
        return None
    sample_path = sample_paths[0]

    import torch

    sample = torch.load(sample_path, map_location="cpu", weights_only=True)
    if getattr(sample, "ndim", None) != 2:
        raise ValueError(
            f"Expected cached site tensor shape (5, D) at {sample_path}, got {tuple(sample.shape)}"
        )
    return int(sample.shape[-1])


def render_startup(
    console: Console,
    *,
    checkpoint_path: str,
    model_name: str,
    feature_root: Path,
    output_profiles_root: Path,
    compressed_root: Path,
    batch: str,
    cell_filter: str | None,
    timelines: list[str],
    total_targets: int,
    downloaded_targets: int,
) -> None:
    """Render the pipeline startup summary."""
    body = "\n".join(
        [
            f"[bold]Checkpoint:[/bold] {checkpoint_path}",
            f"[bold]Backbone:[/bold] {model_name}",
            f"[bold]Compressed root:[/bold] {compressed_root}",
            f"[bold]Feature root:[/bold] {feature_root}",
            f"[bold]Output profiles:[/bold] {output_profiles_root}",
            f"[bold]Batch:[/bold] {batch}",
            f"[bold]Cell filter:[/bold] {cell_filter or 'all'}",
            f"[bold]Timelines:[/bold] {', '.join(timelines)}",
            f"[bold]Benchmark targets:[/bold] {total_targets}",
            f"[bold]Downloaded targets:[/bold] {downloaded_targets}",
        ]
    )
    console.print(Panel.fit(body, title="CellCLIP Plate Pipeline", border_style="blue"))


def render_result_table(console: Console, results: list[PlateResult]) -> None:
    """Render a summary table for all plate results."""
    table = Table(title="Plate Results")
    table.add_column("Plate", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    for result in results:
        style = STATUS_STYLES.get(result.status, "white")
        table.add_row(result.barcode, f"[{style}]{result.status}[/{style}]", result.message)

    console.print(table)


def summarize_results(
    results: list[PlateResult], missing_images: list[str] | None = None
) -> dict[str, int]:
    """Count status totals for the final summary."""
    counts: dict[str, int] = {}
    if missing_images is not None:
        counts["missing_images"] = len(missing_images)
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def run_plate_pipeline(
    *,
    plate: PlatePaths,
    dino_model,
    dino_processor,
    dino_device: str,
    extraction_batch_size: int,
    save_tensors: bool,
    cellclip_model,
    cellclip_device: str,
    source_profiles_root: Path,
    batch: str,
    site_batch_size: int,
    input_dim: int,
    force_export: bool,
    keep_features: bool,
    keep_tensors: bool,
    prune_empty_dirs: bool,
) -> PlateResult:
    """Run extract -> export -> cleanup for one plate barcode."""
    if plate.output_path.exists() and not force_export:
        return PlateResult(
            barcode=plate.barcode,
            status="skipped_existing_export",
            message=f"existing profile at {plate.output_path}",
            output_path=plate.output_path,
        )

    features_complete, extracted_count, expected_count = feature_cache_state(
        plate.feature_dir, plate.image_dir
    )
    features_reused = False
    extraction_note = ""
    cached_width = cached_feature_width(plate.feature_dir)
    cache_width_matches = cached_width is None or cached_width == input_dim

    if features_complete and cache_width_matches:
        features_reused = True
        extraction_note = (
            f"reused {extracted_count}/{expected_count} site features at width {input_dim}"
        )
    else:
        if extracted_count > 0:
            cleared = clear_pt_files(plate.feature_dir)
            if save_tensors:
                clear_pt_files(plate.tensor_dir)
            if features_complete and not cache_width_matches:
                extraction_note = (
                    f"cleared cache with width {cached_width} to enforce width {input_dim}"
                )
            else:
                extraction_note = (
                    f"cleared incomplete cache ({cleared}/{expected_count}) before re-extracting"
                )

        extract_plate_features_with_model(
            image_dir=plate.image_dir,
            output_dir=plate.feature_dir,
            model=dino_model,
            processor=dino_processor,
            device=dino_device,
            batch_size=extraction_batch_size,
            save_tensors=save_tensors,
            tensor_output_dir=plate.tensor_dir if save_tensors else None,
        )
        extraction_note = extraction_note or f"extracted site features at width {input_dim}"

    output_path = export_plate(
        model=cellclip_model,
        device=cellclip_device,
        source_profiles_root=source_profiles_root,
        feature_root=plate.feature_dir.parent,
        output_profiles_root=plate.output_path.parents[2],
        batch=batch,
        plate=plate.barcode,
        site_batch_size=site_batch_size,
    )

    deleted_features = 0
    deleted_tensors = 0
    if not keep_features:
        deleted_features = clear_pt_files(plate.feature_dir)
        if prune_empty_dirs:
            prune_empty_directory(plate.feature_dir)
    if save_tensors and not keep_tensors:
        deleted_tensors = clear_pt_files(plate.tensor_dir)
        if prune_empty_dirs:
            prune_empty_directory(plate.tensor_dir)

    cleanup_parts: list[str] = [extraction_note]
    if not keep_features:
        cleanup_parts.append(f"deleted {deleted_features} feature files")
    if save_tensors and not keep_tensors:
        cleanup_parts.append(f"deleted {deleted_tensors} tensor files")

    return PlateResult(
        barcode=plate.barcode,
        status="exported" if not features_reused else "reused_existing_features",
        message="; ".join(cleanup_parts),
        output_path=output_path,
        features_reused=features_reused,
        features_deleted=deleted_features,
        tensors_deleted=deleted_tensors,
    )
