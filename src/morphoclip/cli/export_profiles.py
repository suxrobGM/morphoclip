"""`morphoclip export-profiles` command: export benchmark-layout profile CSVs.

Encodes well embeddings from a trained MorphoCLIP checkpoint and writes them
in the per-plate CSV layout the benchmark harness (``morphoclip benchmark``,
``benchmark.stable.run_stable_benchmark``) expects.
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from morphoclip.benchmark.export import export_plate_profiles, load_export_models
from morphoclip.cli.logging import setup_logging
from morphoclip.training.inference import discover_plates

console = Console()

DEFAULT_METADATA_CSV = Path("data/reference/cpjump1/cpjump1_metadata.csv")
DEFAULT_OUTPUT_ROOT = Path("data/profiles_morphoclip")
DEFAULT_BATCH = "2020_11_04_CPJUMP1"


def export_profiles(
    checkpoint: Annotated[Path, typer.Option(help="Path to a trained MorphoCLIP checkpoint.")],
    plates: Annotated[
        list[str] | None,
        typer.Option(
            help="Plate barcodes to export (repeatable). Default: all plates under the "
            "checkpoint's feature root."
        ),
    ] = None,
    output_root: Annotated[
        Path, typer.Option(help="Root directory for exported profile CSVs.")
    ] = DEFAULT_OUTPUT_ROOT,
    batch: Annotated[str, typer.Option(help="Batch name used in the output path.")] = DEFAULT_BATCH,
    metadata_csv: Annotated[
        Path, typer.Option(help="Path to the CPJUMP1 reference metadata CSV.")
    ] = DEFAULT_METADATA_CSV,
    negcon_center: Annotated[
        bool,
        typer.Option(
            "--negcon-center/--no-negcon-center",
            help="Center exported features against plate negative controls.",
        ),
    ] = True,
    batch_size: Annotated[int, typer.Option(help="Eval batch size (wells per forward pass).")] = 64,
    device: Annotated[str, typer.Option(help="Device to run inference on.")] = "auto",
) -> None:
    """Export MorphoCLIP well embeddings into benchmark-layout profile CSVs."""
    setup_logging()

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    models = load_export_models(checkpoint, device)
    resolved_plates = plates or discover_plates(Path(models[1].dataset.feature_root))
    if not resolved_plates:
        console.print("[red]No plates to export.[/red]")
        raise typer.Exit(1)

    console.rule("[bold blue]MorphoCLIP Profile Export")
    console.print(
        f"Checkpoint: {checkpoint} | Plates: {len(resolved_plates)} | "
        f"Output root: {output_root} | Batch: {batch} | Negcon center: {negcon_center}"
    )

    table = Table(title="Exported Plate Profiles")
    table.add_column("Plate", style="bold")
    table.add_column("Status")
    table.add_column("Output")

    n_ok, n_failed = 0, 0
    for plate in resolved_plates:
        try:
            output_path = export_plate_profiles(
                checkpoint,
                plate,
                metadata_csv=metadata_csv,
                output_root=output_root,
                batch=batch,
                negcon_center=negcon_center,
                device=device,
                batch_size=batch_size,
                models=models,
            )
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"[red]Plate {plate} failed: {exc}[/red]")
            table.add_row(plate, "[red]failed[/red]", str(exc))
            n_failed += 1
            continue

        console.print(f"[green]Plate {plate}[/green] -> {output_path}")
        table.add_row(plate, "[green]exported[/green]", str(output_path))
        n_ok += 1

    console.print(table)
    console.print(f"\nExported {n_ok}/{len(resolved_plates)} plates ({n_failed} failed).")
    if n_failed:
        raise typer.Exit(1)
