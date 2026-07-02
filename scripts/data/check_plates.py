"""
Scan S3 plate directories and summarize sizes and file counts.
This is a dry-run script to estimate total download size before fetching the dataset.
Requires AWS CLI configured with no-sign-request access to the S3 bucket.
"""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

S3_BASE = "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/"
"""Base S3 URI for the CPJUMP1 images. Plates are subdirectories under this path."""

MAX_WORKERS = 10
"""Maximum number of parallel workers for scanning plates."""


def run_aws(args: list[str]) -> str:
    """Run AWS CLI command and return stdout, or exit on error."""
    result = subprocess.run(
        ["aws", "s3", *args, "--no-sign-request"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        console.log(f"[red]AWS CLI error:[/red] {result.stderr.strip()}")
        sys.exit(1)
    return result.stdout


def list_plates() -> list[str]:
    """List plate directories under the S3 base path."""
    output = run_aws(["ls", S3_BASE])
    plates = []
    for line in output.strip().splitlines():
        parts = line.split()
        if parts and parts[-1].endswith("/"):
            plates.append(parts[-1].rstrip("/"))
    return sorted(plates)


def scan_plate(plate: str) -> tuple[int, int]:
    """Return (total_bytes, file_count) for a plate."""
    output = run_aws(["ls", "--recursive", "--summarize", f"{S3_BASE}{plate}/"])

    total_bytes = 0
    file_count = 0
    for line in output.splitlines():
        if "Total Size:" in line:
            total_bytes = int(line.split("Total Size:")[-1].strip())
        elif "Total Objects:" in line:
            file_count = int(line.split("Total Objects:")[-1].strip())

    return total_bytes, file_count


def format_size(num_bytes: int) -> str:
    """Format byte count into human-readable string."""
    if num_bytes >= 1 << 40:
        return f"{num_bytes / (1 << 40):.2f} TiB"
    if num_bytes >= 1 << 30:
        return f"{num_bytes / (1 << 30):.2f} GiB"
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):.2f} MiB"
    if num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):.2f} KiB"
    return f"{num_bytes} B"


def main() -> None:
    console.log("Scanning plates...")

    plates = list_plates()
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
            futures = {executor.submit(scan_plate, plate): plate for plate in plates}

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
        table.add_row(plate, format_size(size_bytes), str(file_count))

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


if __name__ == "__main__":
    main()
