#!/usr/bin/env python
"""Export benchmark-ready CPJUMP1 profiles using a pretrained CellCLIP model."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cellclip.benchmark import (  # noqa: E402
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CHECKPOINT_REPO,
    export_plate,
    load_cellclip_visual_encoder,
    load_yaml_section,
    normalize_timelines,
    resolve_checkpoint,
    resolve_path,
    select_target_plates,
)
from cellclip.benchmark.export import TIMELINE_CHOICES  # noqa: E402

CONFIG_PATH = Path("configs/benchmark.yml")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Export benchmark-ready CPJUMP1 profiles using pretrained CellCLIP."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--experiment-metadata-path", type=str, default=None)
    parser.add_argument("--source-profiles-root", type=str, default=None)
    parser.add_argument("--feature-root", type=str, default=None)
    parser.add_argument("--output-profiles-root", type=str, default=None)
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
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--checkpoint-repo-id", type=str, default=None)
    parser.add_argument("--checkpoint-filename", type=str, default=None)
    parser.add_argument("--download-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--site-batch-size", type=int, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--input-dim", type=int, default=None)
    parser.add_argument("--loss-type", type=str, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    benchmark_config = load_yaml_section(args.config, "benchmark")
    export_config = load_yaml_section(args.config, "cellclip_export")

    batch = (
        args.batch
        or export_config.get("batch")
        or benchmark_config.get("batch", "2020_11_04_CPJUMP1")
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
    download_dir = resolve_path(args.download_dir, PROJECT_ROOT) if args.download_dir else None

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
    device = args.device or export_config.get("device") or "auto"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    site_batch_size = int(args.site_batch_size or export_config.get("site_batch_size", 16))
    input_dim = int(args.input_dim or export_config.get("input_dim", 1536))
    if args.model_type is not None:
        warnings.warn(
            "--model-type is ignored by the local CellCLIP exporter; "
            "only the visual encoder is loaded.",
            stacklevel=2,
        )
    if args.loss_type is not None:
        warnings.warn(
            "--loss-type is ignored by the local CellCLIP exporter; "
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
        plates = list(dict.fromkeys(args.plates))
    else:
        plates = select_target_plates(
            experiment_metadata_path=experiment_metadata_path,
            batch=batch,
            cell_filter=cell_filter,
            timelines=timelines,
        )

    print("=" * 60)
    print("CellCLIP Profile Export")
    print("=" * 60)
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Input dim:       {input_dim}")
    print(f"Device:          {device}")
    print(f"Feature root:    {feature_root}")
    print(f"Source profiles: {source_profiles_root}")
    print(f"Output profiles: {output_profiles_root}")
    print(f"Batch:           {batch}")
    print(f"Cell filter:     {cell_filter or 'all'}")
    print(f"Timelines:       {', '.join(timelines)}")
    print(f"Plates:          {len(plates)}")
    print("=" * 60)

    model = load_cellclip_visual_encoder(
        model_path=checkpoint_path,
        device=device,
        input_dim=input_dim,
    )

    exported_paths: list[Path] = []
    for index, plate in enumerate(plates, start=1):
        print(f"[{index}/{len(plates)}] Exporting {plate}")
        output_path = export_plate(
            model=model,
            device=device,
            source_profiles_root=source_profiles_root,
            feature_root=feature_root,
            output_profiles_root=output_profiles_root,
            batch=batch,
            plate=plate,
            site_batch_size=site_batch_size,
        )
        exported_paths.append(output_path)
        print(f"Saved: {output_path}")

    print("=" * 60)
    print(f"Export complete: {len(exported_paths)} plate profile files")
    print("=" * 60)


if __name__ == "__main__":
    main()
