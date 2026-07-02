"""`morphoclip cellclip` command group: train, export, pipeline.

CellCLIP is a separate baseline package (`src/cellclip/`). These commands wrap
its local trainer and profile-export pipeline. All relative paths are resolved
against the current working directory (commands are expected to run from the
repository root, like every other MorphoCLIP command).
"""

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from cellclip.benchmark import (
    export_plate,
    load_cellclip_visual_encoder,
    load_yaml_section,
    resolve_export_settings,
    select_target_plates,
)
from cellclip.benchmark.pipeline import (
    PlateResult,
    load_dataset_config,
    render_result_table,
    render_startup,
    resolve_dataset_path,
    resolve_plate_paths,
    run_plate_pipeline,
    summarize_results,
)
from cellclip.training import (
    load_training_config,
    render_train_config,
    render_train_summary,
    train_cellclip,
)
from morphoclip.data.feature_extractor import DEFAULT_MODEL, infer_feature_width, load_dinov3

app = typer.Typer(no_args_is_help=True, help="CellCLIP baseline: train and export profiles.")
console = Console()

DEFAULT_TRAIN_CONFIG = Path("configs/cellclip/cellclip_jumpcp.yaml")
BENCHMARK_CONFIG_PATH = Path("configs/benchmark.yml")
DATASET_CONFIG_PATH = Path("configs/dataset.yml")
CELLCLIP_DEFAULT_MODEL = "facebook/dinov2-giant"


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #


def _resolve_run_dir(config, *, run_name: str | None, output_dir: Path | None) -> Path:
    base_output = output_dir or (Path.cwd() / config.runtime.output_root)
    resolved_name = run_name or config.runtime.run_name
    if resolved_name is None:
        resolved_name = datetime.now().strftime("cellclip_%Y%m%d_%H%M%S")
    return base_output / resolved_name


@app.command()
def train(
    config: Annotated[
        Path, typer.Option(help="CellCLIP training config YAML.")
    ] = DEFAULT_TRAIN_CONFIG,
    run_name: Annotated[str | None, typer.Option(help="Override run name.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Override output root.")] = None,
    split_manifest: Annotated[
        Path | None, typer.Option(help="Override split manifest path.")
    ] = None,
    distributed: Annotated[
        bool, typer.Option(help="Enable DDP multi-GPU training (requires torchrun launcher).")
    ] = False,
) -> None:
    """Train a local CellCLIP model on a CPJUMP1 split subset."""
    cfg = load_training_config(config)
    if distributed:
        cfg.distributed.enabled = True
    if output_dir is not None:
        cfg.runtime.output_root = str(output_dir)
    if run_name is not None:
        cfg.runtime.run_name = run_name
    if split_manifest is not None:
        cfg.dataset.split_manifest_path = str(split_manifest)

    run_dir = _resolve_run_dir(cfg, run_name=run_name, output_dir=output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)

    render_train_config(cfg, config_path=config, run_dir=run_dir)
    result = train_cellclip(cfg, run_dir=run_dir)
    render_train_summary(result)


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


@app.command()
def export(
    config: Annotated[Path, typer.Option(help="Benchmark config YAML.")] = BENCHMARK_CONFIG_PATH,
    experiment_metadata_path: Annotated[str | None, typer.Option()] = None,
    source_profiles_root: Annotated[str | None, typer.Option()] = None,
    feature_root: Annotated[str | None, typer.Option()] = None,
    output_profiles_root: Annotated[str | None, typer.Option()] = None,
    batch: Annotated[str | None, typer.Option()] = None,
    plates: Annotated[
        list[str] | None, typer.Option(help="Restrict to specific plates (repeatable).")
    ] = None,
    cell_filter: Annotated[str | None, typer.Option()] = None,
    timelines: Annotated[
        list[str] | None, typer.Option(help="Timeline labels to export: short and/or long.")
    ] = None,
    ckpt_path: Annotated[str | None, typer.Option()] = None,
    checkpoint_repo_id: Annotated[str | None, typer.Option()] = None,
    checkpoint_filename: Annotated[str | None, typer.Option()] = None,
    download_dir: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option()] = None,
    site_batch_size: Annotated[int | None, typer.Option()] = None,
    model_type: Annotated[str | None, typer.Option(help="Ignored (image-only export).")] = None,
    input_dim: Annotated[int | None, typer.Option()] = None,
    loss_type: Annotated[str | None, typer.Option(help="Ignored (image-only export).")] = None,
) -> None:
    """Export benchmark-ready CPJUMP1 profiles using a pretrained CellCLIP model."""
    project_root = Path.cwd()
    benchmark_config = load_yaml_section(config, "benchmark")
    export_config = load_yaml_section(config, "cellclip_export")

    resolved_batch = (
        batch or export_config.get("batch") or benchmark_config.get("batch", "2020_11_04_CPJUMP1")
    )
    settings = resolve_export_settings(
        project_root=project_root,
        benchmark_config=benchmark_config,
        export_config=export_config,
        context="exporter",
        default_device="auto",
        experiment_metadata_path=experiment_metadata_path,
        source_profiles_root=source_profiles_root,
        feature_root=feature_root,
        output_profiles_root=output_profiles_root,
        cell_filter=cell_filter,
        timelines=timelines,
        ckpt_path=ckpt_path,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_filename=checkpoint_filename,
        download_dir=download_dir,
        device=device,
        site_batch_size=site_batch_size,
        input_dim=input_dim,
        model_type=model_type,
        loss_type=loss_type,
    )

    if plates:
        resolved_plates = list(dict.fromkeys(plates))
    else:
        resolved_plates = select_target_plates(
            experiment_metadata_path=settings.experiment_metadata,
            batch=resolved_batch,
            cell_filter=settings.cell_filter,
            timelines=settings.timelines,
        )

    print("=" * 60)
    print("CellCLIP Profile Export")
    print("=" * 60)
    print(f"Checkpoint:      {settings.checkpoint_path}")
    print(f"Input dim:       {settings.input_dim}")
    print(f"Device:          {settings.device}")
    print(f"Feature root:    {settings.features}")
    print(f"Source profiles: {settings.source_profiles}")
    print(f"Output profiles: {settings.output_profiles}")
    print(f"Batch:           {resolved_batch}")
    print(f"Cell filter:     {settings.cell_filter or 'all'}")
    print(f"Timelines:       {', '.join(settings.timelines)}")
    print(f"Plates:          {len(resolved_plates)}")
    print("=" * 60)

    model = load_cellclip_visual_encoder(
        model_path=settings.checkpoint_path,
        device=settings.device,
        input_dim=settings.input_dim,
    )

    exported_paths: list[Path] = []
    for index, plate in enumerate(resolved_plates, start=1):
        print(f"[{index}/{len(resolved_plates)}] Exporting {plate}")
        output_path = export_plate(
            model=model,
            device=settings.device,
            source_profiles_root=settings.source_profiles,
            feature_root=settings.features,
            output_profiles_root=settings.output_profiles,
            batch=resolved_batch,
            plate=plate,
            site_batch_size=settings.site_batch_size,
        )
        exported_paths.append(output_path)
        print(f"Saved: {output_path}")

    print("=" * 60)
    print(f"Export complete: {len(exported_paths)} plate profile files")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


@app.command()
def pipeline(
    dataset_config: Annotated[Path, typer.Option()] = DATASET_CONFIG_PATH,
    benchmark_config: Annotated[Path, typer.Option()] = BENCHMARK_CONFIG_PATH,
    experiment_metadata_path: Annotated[str | None, typer.Option()] = None,
    source_profiles_root: Annotated[str | None, typer.Option()] = None,
    compressed_root: Annotated[str | None, typer.Option()] = None,
    feature_root: Annotated[str | None, typer.Option()] = None,
    output_profiles_root: Annotated[str | None, typer.Option()] = None,
    tensors_root: Annotated[str | None, typer.Option()] = None,
    batch: Annotated[str | None, typer.Option()] = None,
    plates: Annotated[
        list[str] | None, typer.Option(help="Restrict to specific plates (repeatable).")
    ] = None,
    cell_filter: Annotated[str | None, typer.Option()] = None,
    timelines: Annotated[
        list[str] | None, typer.Option(help="Timeline labels to export: short and/or long.")
    ] = None,
    model_name: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    no_tensors: Annotated[bool, typer.Option()] = False,
    ckpt_path: Annotated[str | None, typer.Option()] = None,
    checkpoint_repo_id: Annotated[str | None, typer.Option()] = None,
    checkpoint_filename: Annotated[str | None, typer.Option()] = None,
    download_dir: Annotated[str | None, typer.Option()] = None,
    site_batch_size: Annotated[int | None, typer.Option()] = None,
    model_type: Annotated[str | None, typer.Option(help="Ignored (image-only export).")] = None,
    input_dim: Annotated[int | None, typer.Option()] = None,
    loss_type: Annotated[str | None, typer.Option(help="Ignored (image-only export).")] = None,
    force_export: Annotated[bool, typer.Option()] = False,
    keep_features: Annotated[bool, typer.Option()] = False,
    keep_tensors: Annotated[bool, typer.Option()] = False,
    prune_empty_dirs: Annotated[bool, typer.Option()] = False,
    stop_on_error: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run extraction, CellCLIP export, and cache cleanup one plate at a time."""
    project_root = Path.cwd()
    ds_config = load_dataset_config(dataset_config)
    bm_config = load_yaml_section(benchmark_config, "benchmark")
    export_config = load_yaml_section(benchmark_config, "cellclip_export")

    resolved_batch = (
        batch or export_config.get("batch") or bm_config.get("batch") or ds_config.get("batch")
    )
    if not resolved_batch:
        raise ValueError("Batch must be provided via --batch or the benchmark/dataset config.")

    settings = resolve_export_settings(
        project_root=project_root,
        benchmark_config=bm_config,
        export_config=export_config,
        context="pipeline",
        default_device=ds_config.get("extraction", {}).get("device", "cuda"),
        experiment_metadata_path=experiment_metadata_path,
        source_profiles_root=source_profiles_root,
        feature_root=feature_root,
        output_profiles_root=output_profiles_root,
        cell_filter=cell_filter,
        timelines=timelines,
        ckpt_path=ckpt_path,
        checkpoint_repo_id=checkpoint_repo_id,
        checkpoint_filename=checkpoint_filename,
        download_dir=download_dir,
        device=device,
        site_batch_size=site_batch_size,
        input_dim=input_dim,
        model_type=model_type,
        loss_type=loss_type,
    )

    compressed = resolve_dataset_path(
        compressed_root
        or ds_config.get("local", {}).get("compressed_images")
        or ds_config.get("compression", {})
        .get("default", {})
        .get("output_root", "data/raw_compressed"),
        project_root,
    )
    tensors = resolve_dataset_path(
        tensors_root or ds_config.get("local", {}).get("tensors", "data/tensors"),
        project_root,
    )

    resolved_model_name = model_name or export_config.get("model_name")
    extraction_batch_size = int(batch_size or ds_config.get("extraction", {}).get("batch_size", 32))
    save_tensors = not no_tensors
    if resolved_model_name is None:
        resolved_model_name = (
            CELLCLIP_DEFAULT_MODEL
            if settings.input_dim == 1536
            else ds_config.get("extraction", {}).get("model", DEFAULT_MODEL)
        )

    if plates:
        target_barcodes = list(dict.fromkeys(plates))
    else:
        target_barcodes = select_target_plates(
            experiment_metadata_path=settings.experiment_metadata,
            batch=resolved_batch,
            cell_filter=settings.cell_filter,
            timelines=settings.timelines,
        )

    plate_paths, missing_images = resolve_plate_paths(
        barcodes=target_barcodes,
        compressed_root=compressed,
        features_root=settings.features,
        tensors_root=tensors,
        output_profiles_root=settings.output_profiles,
        batch=resolved_batch,
    )
    if not plate_paths:
        missing_root = compressed / resolved_batch
        raise RuntimeError(
            f"No target plates have downloaded compressed images under {missing_root}"
        )

    render_startup(
        console,
        checkpoint_path=settings.checkpoint_path,
        model_name=resolved_model_name,
        feature_root=settings.features,
        output_profiles_root=settings.output_profiles,
        compressed_root=compressed,
        batch=resolved_batch,
        cell_filter=settings.cell_filter,
        timelines=settings.timelines,
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
    dino_model, dino_processor = load_dinov3(model_name=resolved_model_name, device=settings.device)
    backbone_width = infer_feature_width(dino_model)
    if backbone_width != settings.input_dim:
        raise RuntimeError(
            f"Vision backbone {resolved_model_name} produces width {backbone_width}, "
            f"but CellCLIP requires --input-dim {settings.input_dim}. "
            "Pick a matching backbone or change --input-dim."
        )
    console.print("[bold]Loading CellCLIP visual encoder...[/bold]")
    cellclip_model = load_cellclip_visual_encoder(
        model_path=settings.checkpoint_path,
        device=settings.device,
        input_dim=settings.input_dim,
    )

    results: list[PlateResult] = list(missing_results)
    for index, plate in enumerate(plate_paths, start=1):
        console.rule(f"[bold blue]{index}/{len(plate_paths)} {plate.barcode}")
        try:
            result = run_plate_pipeline(
                plate=plate,
                dino_model=dino_model,
                dino_processor=dino_processor,
                dino_device=settings.device,
                extraction_batch_size=extraction_batch_size,
                save_tensors=save_tensors,
                cellclip_model=cellclip_model,
                cellclip_device=settings.device,
                source_profiles_root=settings.source_profiles,
                batch=resolved_batch,
                site_batch_size=settings.site_batch_size,
                input_dim=settings.input_dim,
                force_export=force_export,
                keep_features=keep_features,
                keep_tensors=keep_tensors,
                prune_empty_dirs=prune_empty_dirs,
            )
        except Exception as exc:
            result = PlateResult(barcode=plate.barcode, status="failed", message=str(exc))
            console.print(f"[red]Failed:[/red] {plate.barcode}: {exc}")
            if stop_on_error:
                results.append(result)
                break
        else:
            console.print(f"[green]{result.status}[/green]: {result.message}")
        results.append(result)

    render_result_table(console, results)
    counts = summarize_results(results)
    lines = [f"{key}: {value}" for key, value in sorted(counts.items())]
    console.print(Panel.fit("\n".join(lines), title="Summary", border_style="green"))
