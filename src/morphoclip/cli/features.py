"""`morphoclip features` command group: extract and the unattended pipeline.

Upload, download and repack live in `transfer.py` and are registered on this
app at the bottom of this module.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from morphoclip.cli.data import Backend
from morphoclip.cli.transfer import download, repack, upload
from morphoclip.data.config import load_dataset_config
from morphoclip.data.extraction import prepare_plate_output, resolve_image_dir, select_plates
from morphoclip.data.feature_extractor import (
    extract_plate_features,
    verify_plate_features,
)
from morphoclip.data.perturbation import extract_plate_barcode
from morphoclip.data.pipeline import PlateExtractionPipeline
from morphoclip.utils.console import console, setup_logging
from morphoclip.utils.s3 import choose_backend

app = typer.Typer(no_args_is_help=True, help="DINOv3 feature extraction and transfer.")

CONFIG_PATH = Path("configs/dataset.yml")


@app.command()
def extract(
    config: Annotated[Path, typer.Option(help="Dataset config YAML.")] = CONFIG_PATH,
    plate: Annotated[str | None, typer.Option(help="Extract a specific plate only.")] = None,
    model_name: Annotated[
        str | None, typer.Option(help="Override the vision backbone model ID.")
    ] = None,
    compressed_root: Annotated[
        Path | None, typer.Option(help="Override the compressed image root.")
    ] = None,
    features_root: Annotated[
        Path | None, typer.Option(help="Override the feature .pt output root.")
    ] = None,
    tensors_root: Annotated[
        Path | None, typer.Option(help="Override the resized-tensor output root.")
    ] = None,
    verify_only: Annotated[bool, typer.Option(help="Only verify, don't extract.")] = False,
    device: Annotated[str | None, typer.Option(help="Override device (e.g. cuda, cpu).")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override batch size.")] = None,
    no_tensors: Annotated[bool, typer.Option(help="Skip saving resized tensors.")] = False,
) -> None:
    """Extract DINOv3 features from downloaded CPJUMP1 plates."""
    load_dotenv()

    cfg = load_dataset_config(config)

    resolved_model = model_name or cfg.extraction.model
    resolved_device = device or cfg.extraction.device
    resolved_batch_size = batch_size or cfg.extraction.batch_size
    resolved_compressed_root = compressed_root or cfg.local.compressed_images
    resolved_features_root = features_root or cfg.local.features
    resolved_tensors_root = tensors_root or cfg.local.tensors

    plates = select_plates(cfg.plates, plate)

    console.rule("[bold blue]DINOv3 Feature Extraction")
    console.print(f"  Model:      {resolved_model}")
    console.print(f"  Device:     {resolved_device}")
    console.print(f"  Batch size: {resolved_batch_size}")
    console.print(f"  Plates:     {len(plates)}")

    for plate_name in plates:
        barcode = extract_plate_barcode(plate_name)

        image_dir = resolve_image_dir(resolved_compressed_root, cfg.batch, plate_name)
        if image_dir is None:
            missing_dir = resolved_compressed_root / plate_name / "Images"
            console.print(f"\n[bold red]Image directory not found: {missing_dir}")
            continue

        feature_dir = resolved_features_root / barcode
        tensor_dir = resolved_tensors_root / barcode

        if verify_only:
            console.print(f"\n[bold]Verifying [cyan]{barcode}[/cyan]...")
            extracted, expected, missing = verify_plate_features(feature_dir, image_dir)
            console.print(f"  Extracted: {extracted}/{expected}")
            if missing:
                console.print(f"  [red]Missing {len(missing)} sites[/red]")
            else:
                console.print("  [green]All sites extracted[/green]")
            continue

        console.print(f"\n[bold]Processing plate [cyan]{barcode}[/cyan]...")
        console.print(f"  Images:   {image_dir}")
        console.print(f"  Features: {feature_dir}")

        state = prepare_plate_output(image_dir, feature_dir, None if no_tensors else tensor_dir)
        if state.complete:
            console.print(
                "  [yellow]Skipping[/yellow] existing complete batch "
                f"({state.extracted}/{state.expected})"
            )
            continue

        if state.extracted > 0:
            console.print(
                "  [yellow]Incomplete output detected[/yellow] "
                f"({state.extracted}/{state.expected}); clearing and re-extracting batch"
            )
            console.print(f"  Cleared {state.cleared_features} existing feature files")
            if state.cleared_tensors:
                console.print(f"  Cleared {state.cleared_tensors} existing tensor files")

        saved = extract_plate_features(
            image_dir=image_dir,
            output_dir=feature_dir,
            model_name=resolved_model,
            device=resolved_device,
            batch_size=resolved_batch_size,
            save_tensors=not no_tensors,
            tensor_output_dir=tensor_dir if not no_tensors else None,
        )
        console.print(f"  [green]Saved {len(saved)} feature files[/green]")

    console.print("\n[bold green]Done.")


@app.command()
def pipeline(
    config: Annotated[Path, typer.Option(help="Dataset config YAML.")] = CONFIG_PATH,
    progress: Annotated[Path, typer.Option(help="Progress file for crash-safe resume.")] = Path(
        "data/pipeline_progress.json"
    ),
    log_file: Annotated[
        Path | None, typer.Option(help="Log file (default: data/pipeline_{timestamp}.log).")
    ] = None,
    backend: Annotated[Backend | None, typer.Option(help="Transfer backend.")] = None,
    model_name: Annotated[
        str | None, typer.Option(help="Override the vision backbone model ID.")
    ] = None,
    features_root: Annotated[
        Path | None, typer.Option(help="Override the feature .pt output root.")
    ] = None,
    tensors_root: Annotated[
        Path | None, typer.Option(help="Override the resized-tensor output root.")
    ] = None,
    device: Annotated[str | None, typer.Option(help="Override device (e.g. cuda, cpu).")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override batch size.")] = None,
    save_tensors: Annotated[
        bool, typer.Option(help="Also save resized (5,384,384) tensors alongside features.")
    ] = False,
    tensors_only: Annotated[
        bool, typer.Option(help="Save only resized tensors, skip DINOv3 extraction (no GPU).")
    ] = False,
    retry_failed: Annotated[
        bool, typer.Option(help="Reset failed plates to pending and retry them.")
    ] = False,
    plates: Annotated[
        list[str] | None, typer.Option(help="Restrict to specific plate names/barcodes.")
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Log all steps without executing.")] = False,
) -> None:
    """Run the feature extraction pipeline: fetch, extract, clean up.

    Processes plates one at a time, tracking progress for crash-safe resume.
    """
    load_dotenv()

    cfg = load_dataset_config(config)
    if model_name is not None:
        cfg.extraction.model = model_name
    if features_root is not None:
        cfg.local.features = features_root
    if tensors_root is not None:
        cfg.local.tensors = tensors_root

    backend_name = choose_backend(backend.value if backend else cfg.fetch.backend)

    log_path = log_file or Path(f"data/pipeline_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.log")
    setup_logging(log_path=log_path)

    extraction_pipeline = PlateExtractionPipeline(
        config=cfg,
        progress_path=progress,
        backend=backend_name,
        save_tensors=save_tensors,
        tensors_only=tensors_only,
        dry_run=dry_run,
        retry_failed=retry_failed,
    )
    extraction_pipeline.run(device=device, batch_size=batch_size, plates=plates)


app.command()(upload)
app.command()(download)
app.command()(repack)
