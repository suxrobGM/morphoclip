"""`morphoclip features` transfer commands: upload, download, repack.

Moves the pre-extracted `.pt` caches between the local disk and a Hugging Face
dataset repo. Feature extraction itself lives in `features.py`.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import torch
import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

from morphoclip.data.feature_extractor import repack_feature_file
from morphoclip.utils.console import console, make_progress
from morphoclip.utils.hf_features import (
    DEFAULT_REPO_ID,
    download_and_extract_archive,
    list_local_archives,
    list_repo_archives,
    partition_pending_archives,
    upload_folder,
)


def upload(
    features_dir: Annotated[Path, typer.Option(help="Directory of .tar.gz plate archives.")] = Path(
        "data/features_tarred"
    ),
    repo_id: Annotated[str, typer.Option(help="Hugging Face dataset repo ID.")] = DEFAULT_REPO_ID,
    revision: Annotated[
        str | None, typer.Option(help="Branch to upload to (default: main).")
    ] = None,
    num_workers: Annotated[int, typer.Option(help="Number of upload threads.")] = 8,
    dry_run: Annotated[bool, typer.Option(help="List archives without uploading.")] = False,
) -> None:
    """Upload tarred DINOv3 feature archives to a Hugging Face dataset repo."""
    if not features_dir.is_dir():
        console.print(f"[red]Features directory not found: {features_dir}[/red]")
        console.print(
            "[yellow]Run 'uv run poe tar-feature-subfolders --source-dir data/features "
            "--output-dir data/features_tarred' first.[/yellow]"
        )
        raise typer.Exit(1)

    archives = list_local_archives(features_dir)
    if not archives:
        console.print(f"[red]No .tar.gz archives found in {features_dir}[/red]")
        raise typer.Exit(1)

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

    load_dotenv()
    console.print(f"[green]Repository:[/green] https://huggingface.co/datasets/{repo_id}")
    console.print(f"[cyan]Starting upload with {num_workers} workers (resumable)...[/cyan]")

    try:
        upload_folder(features_dir, repo_id=repo_id, revision=revision, num_workers=num_workers)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Re-run to resume automatically.[/yellow]")
        raise typer.Exit(130) from None

    console.print("\n[bold green]Upload complete.[/bold green]")
    console.print(f"https://huggingface.co/datasets/{repo_id}")


def download(
    output_dir: Annotated[Path, typer.Option(help="Directory to extract features into.")] = Path(
        "data/features"
    ),
    repo_id: Annotated[str, typer.Option(help="Hugging Face dataset repo ID.")] = DEFAULT_REPO_ID,
    workers: Annotated[int, typer.Option(help="Number of concurrent download threads.")] = 4,
    skip_extract: Annotated[
        bool, typer.Option(help="Download archives without extracting.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option(help="List available archives without downloading.")
    ] = False,
) -> None:
    """Download and extract DINOv3 feature archives from a Hugging Face dataset repo."""
    load_dotenv()
    api = HfApi()
    archives = list_repo_archives(api, repo_id)
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

    pending, skipped = partition_pending_archives(archives, output_dir)

    if skipped:
        console.print(f"[dim]Skipping {skipped} already-extracted plate(s)[/dim]")
    if not pending:
        console.print("[green]All plates already downloaded and extracted.[/green]")
        return

    console.print(f"[cyan]Downloading {len(pending)} archives with {workers} workers...[/cyan]")
    with make_progress() as progress_bar:
        overall = progress_bar.add_task("[green]Overall", total=len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    download_and_extract_archive,
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
                    progress_bar.console.print(f"  [green]{result}[/green]")
                except Exception as exc:
                    progress_bar.console.print(f"  [red]{archive}: {exc}[/red]")
                progress_bar.advance(overall)

    console.print("\n[bold green]Download complete.[/bold green]")
    console.print(f"Features extracted to: {output_dir}")


def repack(
    feature_root: Annotated[
        Path, typer.Option(help="Root directory holding per-plate feature subdirectories.")
    ] = Path("data/features"),
    plates: Annotated[
        list[str] | None, typer.Option(help="Plate barcodes to repack (default: all).")
    ] = None,
    workers: Annotated[int, typer.Option(help="Number of concurrent repack threads.")] = 8,
    dry_run: Annotated[
        bool, typer.Option(help="Report the projected saving without rewriting files.")
    ] = False,
) -> None:
    """Shrink cached feature files that serialize an oversized backing storage.

    Files saved from an unsliced batch tensor carry the whole batch's storage on
    disk. Rewriting each as a standalone tensor shrinks the cache by roughly
    45x, which makes preloading practical.
    """
    if not feature_root.exists():
        console.print(f"[red]Feature root not found: {feature_root}[/red]")
        raise typer.Exit(1)

    plate_dirs = sorted(d for d in feature_root.iterdir() if d.is_dir())
    if plates:
        wanted = set(plates)
        plate_dirs = [d for d in plate_dirs if d.name in wanted]
    if not plate_dirs:
        console.print("[red]No plate directories to repack.[/red]")
        raise typer.Exit(1)

    files = [f for d in plate_dirs for f in sorted(d.glob("*.pt"))]
    console.print(f"[cyan]Repacking {len(files):,} files across {len(plate_dirs)} plates[/cyan]")

    if dry_run:
        sample = files[: min(50, len(files))]
        before = sum(f.stat().st_size for f in sample)
        after = 0
        for path in sample:
            tensor = torch.load(path, map_location="cpu")
            after += tensor.nbytes
        ratio = before / max(after, 1)
        total = sum(f.stat().st_size for f in files)
        console.print(
            f"[yellow]Dry run:[/yellow] sampled {len(sample)} files, ~{ratio:.1f}x oversized. "
            f"Cache {total / 1024**3:.1f} GiB -> ~{total / ratio / 1024**3:.1f} GiB"
        )
        return

    total_before, total_after = 0, 0
    with make_progress() as progress_bar:
        task = progress_bar.add_task("[green]Repacking", total=len(files))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(repack_feature_file, path): path for path in files}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    before, after = future.result()
                    total_before += before
                    total_after += after
                except Exception as exc:
                    progress_bar.console.print(f"  [red]{path.name}: {exc}[/red]")
                progress_bar.advance(task)

    saved = total_before - total_after
    console.print(
        f"\n[bold green]Repack complete.[/bold green] "
        f"{total_before / 1024**3:.1f} GiB -> {total_after / 1024**3:.1f} GiB "
        f"(freed {saved / 1024**3:.1f} GiB)"
    )
