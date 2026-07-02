"""Fetch CPJUMP1 dataset using AWS CLI or rclone.

Usage examples:
    uv run poe fetch-dataset
    uv run poe fetch-dataset --backend rclone
    uv run poe fetch-dataset --metadata
"""

import argparse
import sys
from pathlib import Path

import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morphoclip.utils.s3 import (  # noqa: E402
    DEFAULT_RCLONE_REMOTE,
    build_s3_uri,
    choose_backend,
    sync_s3_path,
)

CONFIG_PATH = Path("configs/dataset.yml")
console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        print("Please answer 'y' or 'n'.")


# ---------------------------------------------------------------------------
# Per-plate processing
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fetch CPJUMP1 dataset from S3.")
    p.add_argument("--config", type=Path, default=CONFIG_PATH)
    p.add_argument("--metadata", action="store_true", help="Download only metadata.")
    p.add_argument("--backend", choices=["auto", "awscli", "rclone"], default=None)
    p.add_argument("--on-existing-plate", choices=["ask", "skip", "redownload"], default=None)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    with open(args.config) as f:
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

    backend = choose_backend(str(args.backend or fetch_cfg.get("backend", "auto")))
    on_existing_plate = str(args.on_existing_plate or fetch_cfg.get("on_existing_plate", "ask"))

    # --- Summary ---
    console.rule("[bold blue]CPJUMP1 Dataset Fetch")
    console.print(f"  Backend: {backend} | Plates: {len(plates)} | Dry run: {args.dry_run}")

    # --- Metadata ---
    console.print("\n[bold]Downloading plate maps...[/bold]")
    sync_s3_path(
        build_s3_uri(endpoint, cpjump["metadata"], batch),
        metadata_root / "platemaps" / batch,
        backend=backend,
        no_sign_request=no_sign_request,
        rclone_remote=rclone_remote,
        dry_run=args.dry_run,
    )

    console.print("\n[bold]Downloading external metadata...[/bold]")
    sync_s3_path(
        build_s3_uri(endpoint, cpjump["external_metadata"], batch),
        metadata_root / "external_metadata",
        backend=backend,
        no_sign_request=no_sign_request,
        rclone_remote=rclone_remote,
        dry_run=args.dry_run,
    )

    if args.metadata:
        console.print("\n[bold green]Metadata download complete (images skipped).[/bold green]")
        return

    # --- Images ---
    images_uri = build_s3_uri(endpoint, cpjump["images"], batch)
    downloaded = 0

    for plate in plates:
        dl = _process_plate(
            plate,
            plate_uri=f"{images_uri}/{plate}/Images",
            image_dest=raw_root / batch / plate / "Images",
            backend=backend,
            no_sign_request=no_sign_request,
            rclone_remote=rclone_remote,
            dry_run=args.dry_run,
            on_existing_plate=on_existing_plate,
        )
        downloaded += int(dl)

    console.print(f"\n[bold green]Done.[/bold green] Downloaded: {downloaded}/{len(plates)} plates")


if __name__ == "__main__":
    main()
