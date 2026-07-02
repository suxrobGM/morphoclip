#!/usr/bin/env python
"""Run extraction, CellCLIP export, and cache cleanup one plate at a time."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch
from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellclip.benchmark import (  # noqa: E402
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CHECKPOINT_REPO,
    load_cellclip_visual_encoder,
    normalize_timelines,
    resolve_checkpoint,
    select_target_plates,
)
from cellclip.benchmark.export import TIMELINE_CHOICES  # noqa: E402
from cellclip.benchmark.pipeline import (  # noqa: E402
    PlateResult,
    load_dataset_config,
    load_yaml_section,
    render_result_table,
    render_startup,
    resolve_dataset_path,
    resolve_path,
    resolve_plate_paths,
    run_plate_pipeline,
    summarize_results,
)
from morphoclip.data.feature_extractor import (  # noqa: E402
    DEFAULT_MODEL,
    infer_feature_width,
    load_dinov3,
)

console = Console()

DATASET_CONFIG_PATH = Path("configs/dataset.yml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark.yml")
CELLCLIP_DEFAULT_MODEL = "facebook/dinov2-giant"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a disk-bounded CellCLIP pipeline plate by plate."
    )
    parser.add_argument("--dataset-config", type=Path, default=DATASET_CONFIG_PATH)
    parser.add_argument("--benchmark-config", type=Path, default=BENCHMARK_CONFIG_PATH)

    parser.add_argument("--experiment-metadata-path", type=str, default=None)
    parser.add_argument("--source-profiles-root", type=str, default=None)
    parser.add_argument("--compressed-root", type=str, default=None)
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--output-profiles-root", type=str, default=None)
    parser.add_argument("--tensors-root", type=str, default=None)

    parser.add_argument("--batch", type=str, default=None)
    parser.add_argument("--plates", nargs="+", default=None)
    parser.add_argument("--cell-filter", type=str, default=None)
    parser.add_argument(
        "--timelines",
        nargs="+",
        choices=TIMELINE_CHOICES,
        default=None,
        help="Timeline labels to export: short and/or long.",
    )

    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-tensors", action="store_true")

    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--checkpoint-repo-id", type=str, default=None)
    parser.add_argument("--checkpoint-filename", type=str, default=None)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--site-batch-size", type=int, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--loss-type", type=str, default=None)

    parser.add_argument("--force-export", action="store_true")
    parser.add_argument("--keep-features", action="store_true")
    parser.add_argument("--keep-tensors", action="store_true")
    parser.add_argument("--prune-empty-dirs", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    benchmark_config = load_yaml_section(args.benchmark_config, "benchmark")
    export_config = load_yaml_section(args.benchmark_config, "cellclip_export")

    batch = (
        args.batch
        or export_config.get("batch")
        or benchmark_config.get("batch")
        or dataset_config.get("batch")
    )
    timelines = normalize_timelines(
        args.timelines
        if args.timelines is not None
        else export_config.get("timelines", benchmark_config.get("timelines"))
    )
    cell_filter = args.cell_filter or export_config.get("cell_filter")

    experiment_metadata_path = resolve_path(
        args.experiment_metadata_path
        or export_config.get("experiment_metadata_path")
        or "output/benchmark/input/experiment-metadata.tsv",
        PROJECT_ROOT,
    )
    source_profiles_root = resolve_path(
        args.source_profiles_root
        or export_config.get("source_profiles_root")
        or benchmark_config.get("profiles_dir", "data/profiles"),
        PROJECT_ROOT,
    )
    compressed_root = resolve_dataset_path(
        args.compressed_root
        or dataset_config.get("local", {}).get("compressed_images")
        or dataset_config.get("compression", {})
        .get("default", {})
        .get("output_root", "data/raw_compressed"),
        PROJECT_ROOT,
    )
    feature_root = resolve_path(
        args.feature_root or export_config.get("feature_root") or "data/features_cellclip_base",
        PROJECT_ROOT,
    )
    output_profiles_root = resolve_path(
        args.output_profiles_root
        or export_config.get("output_profiles_root")
        or "data/profiles_cellclip_hf",
        PROJECT_ROOT,
    )
    tensors_root = resolve_dataset_path(
        args.tensors_root or dataset_config.get("local", {}).get("tensors", "data/tensors"),
        PROJECT_ROOT,
    )
    download_dir = resolve_path(args.download_dir, PROJECT_ROOT) if args.download_dir else None

    model_name = args.model_name or export_config.get("model_name")
    device = (
        args.device
        or export_config.get("device")
        or dataset_config.get("extraction", {}).get("device", "cuda")
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    extraction_batch_size = int(
        args.batch_size or dataset_config.get("extraction", {}).get("batch_size", 32)
    )
    save_tensors = not args.no_tensors

    checkpoint_repo_id = (
        args.checkpoint_repo_id
        or export_config.get("checkpoint_repo_id")
        or DEFAULT_CHECKPOINT_REPO
    )
    checkpoint_filename = (
        args.checkpoint_filename
        or export_config.get("checkpoint_filename")
        or DEFAULT_CHECKPOINT_FILENAME
    )
    site_batch_size = int(args.site_batch_size or export_config.get("site_batch_size", 16))
    input_dim = int(args.input_dim or export_config.get("input_dim", 1536))
    if model_name is None:
        model_name = (
            CELLCLIP_DEFAULT_MODEL
            if input_dim == 1536
            else dataset_config.get("extraction", {}).get("model", DEFAULT_MODEL)
        )

    if args.model_type is not None:
        warnings.warn(
            "--model-type is ignored by the local CellCLIP pipeline; "
            "only the visual encoder is loaded.",
            stacklevel=2,
        )
    if args.loss_type is not None:
        warnings.warn(
            "--loss-type is ignored by the local CellCLIP pipeline; "
            "it is not needed for image-only export.",
            stacklevel=2,
        )

    checkpoint_path = resolve_checkpoint(
        ckpt_path=args.ckpt_path,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_filename=checkpoint_filename,
        download_dir=download_dir,
    )

    if args.plates:
        target_barcodes = list(dict.fromkeys(args.plates))
    else:
        target_barcodes = select_target_plates(
            experiment_metadata_path=experiment_metadata_path,
            batch=batch,
            cell_filter=cell_filter,
            timelines=timelines,
        )

    plate_paths, missing_images = resolve_plate_paths(
        barcodes=target_barcodes,
        compressed_root=compressed_root,
        features_root=feature_root,
        tensors_root=tensors_root,
        output_profiles_root=output_profiles_root,
        batch=batch,
    )
    if not plate_paths:
        raise RuntimeError(
            f"No target plates have downloaded compressed images under {compressed_root / batch}"
        )

    render_startup(
        console,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        feature_root=feature_root,
        output_profiles_root=output_profiles_root,
        compressed_root=compressed_root,
        batch=batch,
        cell_filter=cell_filter,
        timelines=timelines,
        total_targets=len(target_barcodes),
        downloaded_targets=len(plate_paths),
    )

    missing_results = [
        PlateResult(
            barcode=barcode,
            status="skipped_missing_images",
            message="no downloaded compressed-image plate directory",
        )
        for barcode in missing_images
    ]
    if missing_images:
        console.print(
            Panel.fit(
                "\n".join(missing_images[:10]),
                title=f"Skipping {len(missing_images)} plates with no downloaded images",
                border_style="yellow",
            )
        )

    console.print("[bold]Loading DINO backbone...[/bold]")
    dino_model, dino_processor = load_dinov3(model_name=model_name, device=device)
    backbone_width = infer_feature_width(dino_model)
    if backbone_width != input_dim:
        raise RuntimeError(
            f"Vision backbone {model_name} produces width {backbone_width}, "
            f"but CellCLIP requires --input-dim {input_dim}. "
            "Pick a matching backbone or change --input-dim."
        )
    console.print("[bold]Loading CellCLIP visual encoder...[/bold]")
    cellclip_model = load_cellclip_visual_encoder(
        model_path=checkpoint_path,
        device=device,
        input_dim=input_dim,
    )

    results: list[PlateResult] = list(missing_results)
    for index, plate in enumerate(plate_paths, start=1):
        console.rule(f"[bold blue]{index}/{len(plate_paths)} {plate.barcode}")
        try:
            result = run_plate_pipeline(
                plate=plate,
                dino_model=dino_model,
                dino_processor=dino_processor,
                dino_device=device,
                extraction_batch_size=extraction_batch_size,
                save_tensors=save_tensors,
                cellclip_model=cellclip_model,
                cellclip_device=device,
                source_profiles_root=source_profiles_root,
                batch=batch,
                site_batch_size=site_batch_size,
                input_dim=input_dim,
                force_export=args.force_export,
                keep_features=args.keep_features,
                keep_tensors=args.keep_tensors,
                prune_empty_dirs=args.prune_empty_dirs,
            )
        except Exception as exc:
            result = PlateResult(
                barcode=plate.barcode,
                status="failed",
                message=str(exc),
            )
            console.print(f"[red]Failed:[/red] {plate.barcode}: {exc}")
            if args.stop_on_error:
                results.append(result)
                break
        else:
            console.print(f"[green]{result.status}[/green]: {result.message}")
        results.append(result)

    render_result_table(console, results)
    counts = summarize_results(results)

    lines = [f"{key}: {value}" for key, value in sorted(counts.items())]
    console.print(Panel.fit("\n".join(lines), title="Summary", border_style="green"))


if __name__ == "__main__":
    main()
