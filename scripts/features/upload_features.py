#!/usr/bin/env python3
"""Upload pre-extracted DINOv3 features (tarred) to a Hugging Face dataset repository.

Expects tarred plate archives (one .tar.gz per plate) produced by
scripts/compression/tar_feature_subfolders.py. This keeps the file count
well under HF's 100K recommendation (51 archives instead of ~187K .pt files).

Uses upload_large_folder for resilient, multi-threaded, resumable uploads.
Reads HF_TOKEN from .env file in the project root.

Usage:
    uv run poe upload-features                          # Upload all tarred plates
    uv run poe upload-features --dry-run                # Preview archives without uploading
    uv run poe upload-features --num-workers 16         # Upload with 16 threads
    uv run poe upload-features --repo-id user/my-repo   # Upload to a custom repo
    uv run poe upload-features --revision dev           # Upload to a specific branch
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "features_tarred"
DEFAULT_REPO_ID = "suxrobgm/cpjump1-dinov3-features"

console = Console()


def get_archives(features_dir: Path) -> list[Path]:
    """Return sorted list of .tar.gz archives in the directory."""
    return sorted(features_dir.glob("*.tar.gz"))


def upload_features(
    *,
    features_dir: Path,
    repo_id: str,
    revision: str | None,
    num_workers: int,
    dry_run: bool,
) -> None:
    """Upload the tarred features directory to the HF dataset repo."""
    archives = get_archives(features_dir)
    if not archives:
        console.print(f"[red]No .tar.gz archives found in {features_dir}[/red]")
        console.print(
            "[yellow]Run 'uv run poe tar-feature-subfolders --source-dir data/features "
            "--output-dir data/features_tarred' first.[/yellow]"
        )
        return

    total_size_gb = sum(a.stat().st_size for a in archives) / (1024**3)
    console.print(
        f"[green]Found {len(archives)} archives ({total_size_gb:.1f} GB) to upload[/green]"
    )

    if dry_run:
        for archive in archives:
            size_mb = archive.stat().st_size / (1024**2)
            console.print(f"  [dim]{archive.name}[/dim] — {size_mb:.0f} MB")
        console.print("[yellow]Dry run — no files uploaded.[/yellow]")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    login()

    api = HfApi()

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    console.print(f"[green]Repository:[/green] https://huggingface.co/datasets/{repo_id}")

    console.print(f"[cyan]Starting upload with {num_workers} workers (resumable)...[/cyan]")

    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(features_dir),
        revision=revision,
        num_workers=num_workers,
    )

    console.print("\n[bold green]Upload complete.[/bold green]")
    console.print(f"https://huggingface.co/datasets/{repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload tarred DINOv3 feature archives to Hugging Face.",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=FEATURES_DIR,
        help=f"Directory containing .tar.gz plate archives (default: {FEATURES_DIR})",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch to upload to (default: main)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of upload threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List archives without uploading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.features_dir.is_dir():
        console.print(f"[red]Features directory not found: {args.features_dir}[/red]")
        console.print(
            "[yellow]Run 'uv run poe tar-feature-subfolders --source-dir data/features "
            "--output-dir data/features_tarred' first.[/yellow]"
        )
        return 1

    try:
        upload_features(
            features_dir=args.features_dir,
            repo_id=args.repo_id,
            revision=args.revision,
            num_workers=args.num_workers,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Re-run to resume automatically.[/yellow]")
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
