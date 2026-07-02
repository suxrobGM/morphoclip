"""`morphoclip data` command group: fetch, check-plates."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from morphoclip.utils.s3 import (
    DEFAULT_RCLONE_REMOTE,
    build_s3_uri,
    choose_backend,
    sync_s3_path,
)

app = typer.Typer(no_args_is_help=True, help="Dataset fetching and inspection.")
console = Console()

CONFIG_PATH = Path("configs/dataset.yml")


class Backend(StrEnum):
    auto = "auto"
    awscli = "awscli"
    rclone = "rclone"


class OnExisting(StrEnum):
    ask = "ask"
    skip = "skip"
    redownload = "redownload"


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #


def _count_tiffs(image_dir: Path) -> int:
    """Count TIFF files in a directory."""
    return len(list(image_dir.glob("*.tif"))) + len(list(image_dir.glob("*.tiff")))


def _prompt_existing(label: str, dest: Path, action_yes: str, mode: str) -> str:
    """Ask the user what to do when files already exist at *dest*.

    Returns *action_yes* (e.g. 'redownload') or 'skip'.
    """
    if mode == action_yes:
        return action_yes
    if mode == "skip":
        return "skip"

    if not sys.stdin.isatty():
        console.print(
            f"  [yellow]{label} already exists at {dest}, non-interactive -> skip.[/yellow]"
        )
        return "skip"

    while True:
        answer = input(f"{label} already exists at {dest}. Redo? [y/N]: ").strip().lower()
        if answer in {"y", "yes"}:
            return action_yes
        if answer in {"", "n", "no"}:
            return "skip"
        console.print("Please answer 'y' or 'n'.")


def _process_plate(
    plate: str,
    *,
    plate_uri: str,
    image_dest: Path,
    backend: str,
    no_sign_request: bool,
    rclone_remote: str,
    dry_run: bool,
    on_existing_plate: str,
) -> bool:
    """Handle download for one plate. Returns True if downloaded."""
    should_download = True
    if image_dest.exists():
        existing_tiffs = _count_tiffs(image_dest)
        if existing_tiffs > 0:
            action = _prompt_existing(
                f"Plate {plate} ({existing_tiffs} TIFFs)",
                image_dest,
                "redownload",
                on_existing_plate,
            )
            should_download = action == "redownload"
            verb = "Re-downloading" if should_download else "Skipping download for"
            console.print(f"  [bold]{verb} [cyan]{plate}[/cyan].[/bold]")
    else:
        console.print(f"  [bold]Downloading plate [cyan]{plate}[/cyan]...[/bold]")

    if should_download:
        sync_s3_path(
            plate_uri,
            image_dest,
            backend=backend,
            no_sign_request=no_sign_request,
            rclone_remote=rclone_remote,
            dry_run=dry_run,
        )

    return should_download


@app.command()
def fetch(
    config: Annotated[Path, typer.Option(help="Dataset config YAML.")] = CONFIG_PATH,
    metadata: Annotated[bool, typer.Option(help="Download only metadata (skip images).")] = False,
    backend: Annotated[
        Backend | None, typer.Option(help="Transfer backend (default: from config).")
    ] = None,
    on_existing_plate: Annotated[
        OnExisting | None,
        typer.Option(help="What to do when a plate already exists (default: from config)."),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Log actions without transferring.")] = False,
) -> None:
    """Fetch the CPJUMP1 dataset from S3 (AWS CLI or rclone)."""
    with open(config) as f:
        cpjump = yaml.safe_load(f)["cpjump"]

    endpoint = cpjump["endpoint"]
    batch = cpjump["batch"]
    plates = cpjump["plates"]

    local = cpjump.get("local", {})
    raw_root = Path(local.get("raw_images", "data/raw"))
    metadata_root = Path(local.get("metadata", "data/metadata"))

    fetch_cfg = cpjump.get("fetch", {})
    no_sign_request = bool(fetch_cfg.get("aws_no_sign_request", True))
    rclone_remote = str(fetch_cfg.get("rclone_remote", DEFAULT_RCLONE_REMOTE))

    backend_name = choose_backend(
        str(backend.value if backend else fetch_cfg.get("backend", "auto"))
    )
    on_existing = str(
        on_existing_plate.value if on_existing_plate else fetch_cfg.get("on_existing_plate", "ask")
    )

    console.rule("[bold blue]CPJUMP1 Dataset Fetch")
    console.print(f"  Backend: {backend_name} | Plates: {len(plates)} | Dry run: {dry_run}")

    console.print("\n[bold]Downloading plate maps...[/bold]")
    sync_s3_path(
        build_s3_uri(endpoint, cpjump["metadata"], batch),
        metadata_root / "platemaps" / batch,
        backend=backend_name,
        no_sign_request=no_sign_request,
        rclone_remote=rclone_remote,
        dry_run=dry_run,
    )

    console.print("\n[bold]Downloading external metadata...[/bold]")
    sync_s3_path(
        build_s3_uri(endpoint, cpjump["external_metadata"], batch),
        metadata_root / "external_metadata",
        backend=backend_name,
        no_sign_request=no_sign_request,
        rclone_remote=rclone_remote,
        dry_run=dry_run,
    )

    if metadata:
        console.print("\n[bold green]Metadata download complete (images skipped).[/bold green]")
        return

    images_uri = build_s3_uri(endpoint, cpjump["images"], batch)
    downloaded = 0
    for plate in plates:
        dl = _process_plate(
            plate,
            plate_uri=f"{images_uri}/{plate}/Images",
            image_dest=raw_root / batch / plate / "Images",
            backend=backend_name,
            no_sign_request=no_sign_request,
            rclone_remote=rclone_remote,
            dry_run=dry_run,
            on_existing_plate=on_existing,
        )
        downloaded += int(dl)

    console.print(f"\n[bold green]Done.[/bold green] Downloaded: {downloaded}/{len(plates)} plates")


# --------------------------------------------------------------------------- #
# check-plates
# --------------------------------------------------------------------------- #

S3_BASE = "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
"""Base S3 URI for the CPJUMP1 images. Plates are subdirectories under this path."""

MAX_WORKERS = 10
"""Maximum number of parallel workers for scanning plates."""


def _run_aws(args: list[str]) -> str:
    """Run an AWS CLI command and return stdout, or exit on error."""
    result = subprocess.run(
        ["aws", "s3", *args, "--no-sign-request"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.log(f"[red]AWS CLI error:[/red] {result.stderr.strip()}")
        raise typer.Exit(1)
    return result.stdout


def _list_plates() -> list[str]:
    """List plate directories under the S3 base path."""
    output = _run_aws(["ls", S3_BASE])
    plates = []
    for line in output.strip().splitlines():
        parts = line.split()
        if parts and parts[-1].endswith("/"):
            plates.append(parts[-1].rstrip("/"))
    return sorted(plates)


def _scan_plate(plate: str) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a plate."""
    output = _run_aws(["ls", "--recursive", "--summarize", f"{S3_BASE}{plate}/"])
    total_bytes = 0
    file_count = 0
    for line in output.splitlines():
        if "Total Size:" in line:
            total_bytes = int(line.split("Total Size:")[-1].strip())
        elif "Total Objects:" in line:
            file_count = int(line.split("Total Objects:")[-1].strip())
    return total_bytes, file_count


def _format_size(num_bytes: int) -> str:
    """Format a byte count into a human-readable string."""
    if num_bytes >= 1 << 40:
        return f"{num_bytes / (1 << 40):.2f} TiB"
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GiB"
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MiB"
    if num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.2f} KiB"
    return f"{num_bytes} B"


@app.command("check-plates")
def check_plates() -> None:
    """Scan S3 plate directories and summarize sizes/file counts (dry-run estimate)."""
    console.log("Scanning plates...")

    plates = _list_plates()
    console.log(
        f"Found [bold]{len(plates)}[/bold] plates, scanning in parallel (workers={MAX_WORKERS})..."
    )

    results: dict[str, tuple[int, int]] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Scanning plates", total=len(plates))
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_scan_plate, plate): plate for plate in plates}
            for future in as_completed(futures):
                plate = futures[future]
                size_bytes, file_count = future.result()
                results[plate] = (size_bytes, file_count)
                progress.update(task, advance=1, description=f"Scanned [cyan]{plate}[/cyan]")

    total_bytes = sum(size for size, _ in results.values())
    total_count = sum(count for _, count in results.values())
    total_gib = total_bytes / (1 << 30)
    total_tib = total_bytes / (1 << 40)

    table = Table(title="Plate Size Summary")
    table.add_column("Plate", style="cyan", min_width=40)
    table.add_column("Size", justify="right", style="green")
    table.add_column("Files", justify="right", style="yellow")
    for plate in sorted(results):
        size_bytes, file_count = results[plate]
        table.add_row(plate, _format_size(size_bytes), str(file_count))
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]", f"[bold]{total_gib:.2f} GiB[/bold]", f"[bold]{total_count}[/bold]"
    )

    console.print()
    console.print(table)
    console.print()
    console.log(f"Total plates: [bold]{len(results)}[/bold]")
    console.log(f"Total size:   [bold]{total_gib:.2f} GiB ({total_tib:.2f} TiB)[/bold]")
    console.log(f"Total files:  [bold]{total_count}[/bold]")
