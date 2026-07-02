#!/usr/bin/env python3
"""Download and extract DINOv3 features from Hugging Face.

Downloads tarred plate archives from the HF dataset repo and extracts them
into the local features directory. Uses concurrent threads for parallel
downloads and extraction.

Usage:
    uv run poe download-features                          # Download and extract all plates
    uv run poe download-features --dry-run                # Preview available archives
    uv run poe download-features --workers 16             # Download with 16 threads
    uv run poe download-features --repo-id user/my-repo   # Download from a custom repo
    uv run poe download-features --skip-extract           # Download only, don't extract
"""

import argparse
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
DEFAULT_REPO_ID = "suxrobgm/cpjump1-dinov3-features"

console = Console()


def list_archives(api: HfApi, repo_id: str) -> list[str]:
    """List all .tar.gz files in the HF dataset repo."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    return sorted(f for f in files if f.endswith(".tar.gz") and not f.startswith("."))


def download_and_extract(
    *,
    api: HfApi,
    repo_id: str,
    filename: str,
    output_dir: Path,
    skip_extract: bool,
) -> str:
    """Download a single archive and extract it. Returns the plate name."""
    plate_name = filename.removesuffix(".tar.gz")
    plate_dir = output_dir / plate_name

    # Skip if already extracted
    if plate_dir.is_dir() and any(plate_dir.glob("*.pt")):
        return f"{plate_name} (skipped, already extracted)"

    # Download to HF cache and get local path
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
    )

    if skip_extract:
        return f"{plate_name} (downloaded)"

    # Extract tar.gz to output directory
    with tarfile.open(local_path, "r:gz") as tar:
        tar.extractall(path=output_dir)

    return f"{plate_name} (done)"


def download_features(
    *,
    output_dir: Path,
    repo_id: str,
    workers: int,
    skip_extract: bool,
    dry_run: bool,
) -> None:
    """Download and extract all feature archives from the HF dataset repo."""
    load_dotenv(PROJECT_ROOT / ".env")

    api = HfApi()
    archives = list_archives(api, repo_id)

    if not archives:
        console.print(f"[red]No .tar.gz archives found in {repo_id}[/red]")
        return

    console.print(f"[green]Found {len(archives)} archives in {repo_id}[/green]")

    if dry_run:
        for archive in archives:
            console.print(f"  [dim]{archive}[/dim]")
        console.print("[yellow]Dry run — no files downloaded.[/yellow]")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out already-extracted plates
    pending = []
    skipped = 0
    for archive in archives:
        plate_name = archive.removesuffix(".tar.gz")
        plate_dir = output_dir / plate_name
        if plate_dir.is_dir() and any(plate_dir.glob("*.pt")):
            skipped += 1
        else:
            pending.append(archive)

    if skipped:
        console.print(f"[dim]Skipping {skipped} already-extracted plate(s)[/dim]")

    if not pending:
        console.print("[green]All plates already downloaded and extracted.[/green]")
        return

    console.print(f"[cyan]Downloading {len(pending)} archives with {workers} workers...[/cyan]")

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("[green]Overall", total=len(pending))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    download_and_extract,
                    api=api,
                    repo_id=repo_id,
                    filename=archive,
                    output_dir=output_dir,
                    skip_extract=skip_extract,
                ): archive
                for archive in pending
            }

            for future in as_completed(futures):
                archive = futures[future]
                try:
                    result = future.result()
                    progress.console.print(f"  [green]{result}[/green]")
                except Exception as exc:
                    progress.console.print(f"  [red]{archive}: {exc}[/red]")
                progress.advance(overall)

    console.print("\n[bold green]Download complete.[/bold green]")
    console.print(f"Features extracted to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract DINOv3 features from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FEATURES_DIR,
        help=f"Directory to extract features into (default: {FEATURES_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Download archives without extracting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List available archives without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        download_features(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            workers=args.workers,
            skip_extract=args.skip_extract,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        console.print(
            "\n[yellow]Interrupted. Re-run to resume "
            "(already-extracted plates are skipped).[/yellow]"
        )
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
