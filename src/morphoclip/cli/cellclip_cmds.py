"""`morphoclip cellclip` command group: train, export, pipeline.

CellCLIP is a separate baseline package (`src/cellclip/`). These commands wrap
its local trainer and profile-export pipeline. All relative paths are resolved
against the current working directory (commands are expected to run from the
repository root, like every other MorphoCLIP command).
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Annotated

import torch
import typer
import yaml
from rich.console import Console
from rich.panel import Panel

from cellclip.benchmark import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_CHECKPOINT_REPO,
    export_plate,
    load_cellclip_visual_encoder,
    normalize_timelines,
    resolve_checkpoint,
    select_target_plates,
)
from cellclip.benchmark import load_yaml_section as load_benchmark_yaml_section
from cellclip.benchmark import resolve_path as resolve_export_path
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
from cellclip.benchmark.pipeline import load_yaml_section as load_pipeline_yaml_section
from cellclip.benchmark.pipeline import resolve_path as resolve_pipeline_path
from cellclip.training import load_training_config, train_cellclip
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

    print("=" * 60)
    print("Local CellCLIP Training")
    print("=" * 60)
    print(f"Config:            {config}")
    print(f"Run directory:     {run_dir}")
    print(f"Feature root:      {cfg.dataset.feature_root}")
    print(f"Split strategy:    {cfg.dataset.split_strategy}")
    print(f"Train subset:      {cfg.dataset.subset}")
    print(f"Eval subset:       {cfg.dataset.eval_subset}")
    print(f"Unique perts:      {cfg.dataset.unique_perturbations}")
    print(f"Train max sites:   {cfg.dataset.train_max_sites_per_well}")
    print(f"Eval max sites:    {cfg.dataset.eval_max_sites_per_well}")
    print(f"Within-well interp:{cfg.dataset.within_well_interp_sites}")
    print(f"Same-pert interp:  {cfg.dataset.same_pert_interp_sites}")
    print(f"Interp alpha:      {cfg.dataset.interp_alpha}")
    print(f"Model variant:     {cfg.model.variant}")
    print(f"Text model:        {cfg.model.text_model_name}")
    print(f"Tokenizer:         {cfg.model.tokenizer_name}")
    if cfg.model.variant in {"chemberta_film", "chemberta"}:
        print(f"ChemBERTa model:   {cfg.model.chemberta_model_name}")
        print(f"SMILES tokenizer:  {cfg.model.chemberta_tokenizer_name}")
        print(f"Chem fusion:       {cfg.model.chem_fusion_type}")
        print(f"Prompt policy:     {cfg.model.chem_prompt_policy}")
        print(f"Chem pooling:      {cfg.model.chemberta_pooling}")
        print(f"Freeze ChemBERTa:  {cfg.model.freeze_chemberta}")
        print(f"Tune top layers:   {cfg.model.chemberta_tune_layers}")
    print(f"Loss:              {cfg.optimization.loss_type}")
    print("=" * 60)

    result = train_cellclip(cfg, run_dir=run_dir)

    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Train wells:       {result['train_wells']}")
    print(f"Eval wells:        {result['eval_wells']}")
    print(f"Metrics:           {result['metrics_path']}")
    print(f"Best checkpoint:   {result['best_checkpoint']}")
    print(f"Last checkpoint:   {result['last_checkpoint']}")
    print("=" * 60)


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
    benchmark_config = load_benchmark_yaml_section(config, "benchmark")
    export_config = load_benchmark_yaml_section(config, "cellclip_export")

    resolved_batch = (
        batch or export_config.get("batch") or benchmark_config.get("batch", "2020_11_04_CPJUMP1")
    )
    resolved_timelines = normalize_timelines(
        timelines
        if timelines is not None
        else export_config.get("timelines", benchmark_config.get("timelines"))
    )
    resolved_cell_filter = cell_filter or export_config.get("cell_filter")

    experiment_metadata = resolve_export_path(
        experiment_metadata_path
        or export_config.get("experiment_metadata_path")
        or "output/benchmark/input/experiment-metadata.tsv",
        project_root,
    )
    source_profiles = resolve_export_path(
        source_profiles_root
        or export_config.get("source_profiles_root")
        or benchmark_config.get("profiles_dir", "data/profiles"),
        project_root,
    )
    features = resolve_export_path(
        feature_root or export_config.get("feature_root") or "data/features_cellclip_base",
        project_root,
    )
    output_profiles = resolve_export_path(
        output_profiles_root
        or export_config.get("output_profiles_root")
        or "data/profiles_cellclip_hf",
        project_root,
    )
    resolved_download_dir = (
        resolve_export_path(download_dir, project_root) if download_dir else None
    )

    resolved_repo_id = (
        checkpoint_repo_id or export_config.get("checkpoint_repo_id") or DEFAULT_CHECKPOINT_REPO
    )
    resolved_filename = (
        checkpoint_filename
        or export_config.get("checkpoint_filename")
        or DEFAULT_CHECKPOINT_FILENAME
    )
    resolved_device = device or export_config.get("device") or "auto"
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_site_batch_size = int(site_batch_size or export_config.get("site_batch_size", 16))
    resolved_input_dim = int(input_dim or export_config.get("input_dim", 1536))
    if model_type is not None:
        warnings.warn(
            "--model-type is ignored by the local CellCLIP exporter; "
            "only the visual encoder is loaded.",
            stacklevel=2,
        )
    if loss_type is not None:
        warnings.warn(
            "--loss-type is ignored by the local CellCLIP exporter; "
            "it is not needed for image-only export.",
            stacklevel=2,
        )

    checkpoint_path = resolve_checkpoint(
        ckpt_path=ckpt_path,
        checkpoint_repo_id=resolved_repo_id,
        checkpoint_filename=resolved_filename,
        download_dir=resolved_download_dir,
    )

    if plates:
        resolved_plates = list(dict.fromkeys(plates))
    else:
        resolved_plates = select_target_plates(
            experiment_metadata_path=experiment_metadata,
            batch=resolved_batch,
            cell_filter=resolved_cell_filter,
            timelines=resolved_timelines,
        )

    print("=" * 60)
    print("CellCLIP Profile Export")
    print("=" * 60)
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Input dim:       {resolved_input_dim}")
    print(f"Device:          {resolved_device}")
    print(f"Feature root:    {features}")
    print(f"Source profiles: {source_profiles}")
    print(f"Output profiles: {output_profiles}")
    print(f"Batch:           {resolved_batch}")
    print(f"Cell filter:     {resolved_cell_filter or 'all'}")
    print(f"Timelines:       {', '.join(resolved_timelines)}")
    print(f"Plates:          {len(resolved_plates)}")
    print("=" * 60)

    model = load_cellclip_visual_encoder(
        model_path=checkpoint_path,
        device=resolved_device,
        input_dim=resolved_input_dim,
    )

    exported_paths: list[Path] = []
    for index, plate in enumerate(resolved_plates, start=1):
        print(f"[{index}/{len(resolved_plates)}] Exporting {plate}")
        output_path = export_plate(
            model=model,
            device=resolved_device,
            source_profiles_root=source_profiles,
            feature_root=features,
            output_profiles_root=output_profiles,
            batch=resolved_batch,
            plate=plate,
            site_batch_size=resolved_site_batch_size,
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
    bm_config = load_pipeline_yaml_section(benchmark_config, "benchmark")
    export_config = load_pipeline_yaml_section(benchmark_config, "cellclip_export")

    resolved_batch = (
        batch or export_config.get("batch") or bm_config.get("batch") or ds_config.get("batch")
    )
    if not resolved_batch:
        raise ValueError(
            "Batch must be provided via --batch or the benchmark/dataset config."
        )
    resolved_timelines = normalize_timelines(
        timelines
        if timelines is not None
        else export_config.get("timelines", bm_config.get("timelines"))
    )
    resolved_cell_filter = cell_filter or export_config.get("cell_filter")

    experiment_metadata = resolve_pipeline_path(
        experiment_metadata_path
        or export_config.get("experiment_metadata_path")
        or "output/benchmark/input/experiment-metadata.tsv",
        project_root,
    )
    source_profiles = resolve_pipeline_path(
        source_profiles_root
        or export_config.get("source_profiles_root")
        or bm_config.get("profiles_dir", "data/profiles"),
        project_root,
    )
    compressed = resolve_dataset_path(
        compressed_root
        or ds_config.get("local", {}).get("compressed_images")
        or ds_config.get("compression", {})
        .get("default", {})
        .get("output_root", "data/raw_compressed"),
        project_root,
    )
    features = resolve_pipeline_path(
        feature_root or export_config.get("feature_root") or "data/features_cellclip_base",
        project_root,
    )
    output_profiles = resolve_pipeline_path(
        output_profiles_root
        or export_config.get("output_profiles_root")
        or "data/profiles_cellclip_hf",
        project_root,
    )
    tensors = resolve_dataset_path(
        tensors_root or ds_config.get("local", {}).get("tensors", "data/tensors"),
        project_root,
    )
    resolved_download_dir = (
        resolve_pipeline_path(download_dir, project_root) if download_dir else None
    )

    resolved_model_name = model_name or export_config.get("model_name")
    resolved_device = (
        device
        or export_config.get("device")
        or ds_config.get("extraction", {}).get("device", "cuda")
    )
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    extraction_batch_size = int(batch_size or ds_config.get("extraction", {}).get("batch_size", 32))
    save_tensors = not no_tensors

    resolved_repo_id = (
        checkpoint_repo_id or export_config.get("checkpoint_repo_id") or DEFAULT_CHECKPOINT_REPO
    )
    resolved_filename = (
        checkpoint_filename
        or export_config.get("checkpoint_filename")
        or DEFAULT_CHECKPOINT_FILENAME
    )
    resolved_site_batch_size = int(site_batch_size or export_config.get("site_batch_size", 16))
    resolved_input_dim = int(input_dim or export_config.get("input_dim", 1536))
    if resolved_model_name is None:
        resolved_model_name = (
            CELLCLIP_DEFAULT_MODEL
            if resolved_input_dim == 1536
            else ds_config.get("extraction", {}).get("model", DEFAULT_MODEL)
        )

    if model_type is not None:
        warnings.warn(
            "--model-type is ignored by the local CellCLIP pipeline; "
            "only the visual encoder is loaded.",
            stacklevel=2,
        )
    if loss_type is not None:
        warnings.warn(
            "--loss-type is ignored by the local CellCLIP pipeline; "
            "it is not needed for image-only export.",
            stacklevel=2,
        )

    checkpoint_path = resolve_checkpoint(
        ckpt_path=ckpt_path,
        checkpoint_repo_id=resolved_repo_id,
        checkpoint_filename=resolved_filename,
        download_dir=resolved_download_dir,
    )

    if plates:
        target_barcodes = list(dict.fromkeys(plates))
    else:
        target_barcodes = select_target_plates(
            experiment_metadata_path=experiment_metadata,
            batch=resolved_batch,
            cell_filter=resolved_cell_filter,
            timelines=resolved_timelines,
        )

    plate_paths, missing_images = resolve_plate_paths(
        barcodes=target_barcodes,
        compressed_root=compressed,
        features_root=features,
        tensors_root=tensors,
        output_profiles_root=output_profiles,
        batch=resolved_batch,
    )
    if not plate_paths:
        missing_root = compressed / resolved_batch
        raise RuntimeError(
            f"No target plates have downloaded compressed images under {missing_root}"
        )

    render_startup(
        console,
        checkpoint_path=checkpoint_path,
        model_name=resolved_model_name,
        feature_root=features,
        output_profiles_root=output_profiles,
        compressed_root=compressed,
        batch=resolved_batch,
        cell_filter=resolved_cell_filter,
        timelines=resolved_timelines,
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
    dino_model, dino_processor = load_dinov3(model_name=resolved_model_name, device=resolved_device)
    backbone_width = infer_feature_width(dino_model)
    if backbone_width != resolved_input_dim:
        raise RuntimeError(
            f"Vision backbone {resolved_model_name} produces width {backbone_width}, "
            f"but CellCLIP requires --input-dim {resolved_input_dim}. "
            "Pick a matching backbone or change --input-dim."
        )
    console.print("[bold]Loading CellCLIP visual encoder...[/bold]")
    cellclip_model = load_cellclip_visual_encoder(
        model_path=checkpoint_path,
        device=resolved_device,
        input_dim=resolved_input_dim,
    )

    results: list[PlateResult] = list(missing_results)
    for index, plate in enumerate(plate_paths, start=1):
        console.rule(f"[bold blue]{index}/{len(plate_paths)} {plate.barcode}")
        try:
            result = run_plate_pipeline(
                plate=plate,
                dino_model=dino_model,
                dino_processor=dino_processor,
                dino_device=resolved_device,
                extraction_batch_size=extraction_batch_size,
                save_tensors=save_tensors,
                cellclip_model=cellclip_model,
                cellclip_device=resolved_device,
                source_profiles_root=source_profiles,
                batch=resolved_batch,
                site_batch_size=resolved_site_batch_size,
                input_dim=resolved_input_dim,
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
