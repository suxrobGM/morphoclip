"""`morphoclip features` command group: extract, pipeline, upload, download."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import torch
import typer
import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from morphoclip.cli.data import Backend
from morphoclip.data.feature_extractor import extract_plate_features, verify_plate_features
from morphoclip.data.image_loader import discover_sites, load_site_as_tensor
from morphoclip.data.perturbation import extract_plate_barcode
from morphoclip.data.pipeline import PlateExtractionPipeline
from morphoclip.data.pipeline import setup_logging as setup_pipeline_logging
from morphoclip.utils.hf_features import (
    DEFAULT_REPO_ID,
    download_and_extract_archive,
    list_local_archives,
    list_repo_archives,
    partition_pending_archives,
    upload_folder,
)
from morphoclip.utils.s3 import choose_backend

app = typer.Typer(no_args_is_help=True, help="DINOv3 feature extraction and transfer.")
console = Console()

CONFIG_PATH = Path("configs/dataset.yml")


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #


def _clear_pt_files(directory: Path) -> int:
    """Remove saved ``.pt`` files from a directory if it exists."""
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.glob("*.pt"):
        path.unlink()
        removed += 1
    return removed


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
    visualize: Annotated[
        bool, typer.Option(help="Save channel grid and composite PNGs for sample sites.")
    ] = False,
    visualize_n: Annotated[int, typer.Option(help="Sample sites to visualize per plate.")] = 4,
    visualize_only: Annotated[
        bool, typer.Option(help="Only generate visualizations, skip extraction.")
    ] = False,
) -> None:
    """Extract DINOv3 features from downloaded CPJUMP1 plates."""
    load_dotenv()

    with open(config) as f:
        cfg = yaml.safe_load(f)["cpjump"]

    extraction = cfg.get("extraction", {})
    local = cfg.get("local", {})
    resolved_model = model_name or extraction.get(
        "model", "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    resolved_device = device or extraction.get("device", "auto")
    resolved_batch_size = batch_size or extraction.get("batch_size", 32)

    resolved_compressed_root = compressed_root or Path(
        local.get(
            "compressed_images",
            cfg.get("compression", {}).get("default", {}).get("output_root", "data/raw_compressed"),
        )
    )
    resolved_features_root = features_root or Path(local.get("features", "data/features"))
    resolved_tensors_root = tensors_root or Path(local.get("tensors", "data/tensors"))

    plates = cfg.get("plates", [])
    if plate:
        plates = [p for p in plates if extract_plate_barcode(p) == plate or p == plate]
        if not plates:
            plates = [plate]

    console.rule("[bold blue]DINOv3 Feature Extraction")
    console.print(f"  Model:      {resolved_model}")
    console.print(f"  Device:     {resolved_device}")
    console.print(f"  Batch size: {resolved_batch_size}")
    console.print(f"  Plates:     {len(plates)}")

    batch = cfg.get("batch", "")
    for plate_name in plates:
        barcode = extract_plate_barcode(plate_name)

        image_dir = resolved_compressed_root / batch / plate_name / "Images"
        if not image_dir.exists():
            image_dir = resolved_compressed_root / plate_name / "Images"
        if not image_dir.exists():
            console.print(f"\n[bold red]Image directory not found: {image_dir}")
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

        if visualize or visualize_only:
            # NOTE: `morphoclip.data.visualize` is not present in the package; this
            # path mirrors the original (broken) script and errors only if used.
            from morphoclip.data.visualize import save_site_comparison

            vis_dir = Path("data/visualizations") / barcode
            sites = discover_sites(image_dir)
            sample_keys = sorted(sites.keys(), key=str)[:visualize_n]
            console.print(
                f"\n[bold]Visualizing [cyan]{barcode}[/cyan] ({len(sample_keys)} sample sites)..."
            )
            for key in sample_keys:
                site_tensor = load_site_as_tensor(sites[key], resize=384)
                feat_path = feature_dir / f"r{key.row:02d}c{key.col:02d}f{key.field:02d}.pt"
                cls_features = None
                if feat_path.exists():
                    cls_features = torch.load(feat_path, weights_only=True)
                cmp_path = save_site_comparison(
                    site_tensor, key, vis_dir, cls_features=cls_features
                )
                console.print(f"  {key}: [dim]{cmp_path}[/dim]")
                if cls_features is not None:
                    console.print(f"         [dim]CLS features: {tuple(cls_features.shape)}[/dim]")
                else:
                    console.print(
                        "         [yellow]No CLS features found (run extraction first)[/yellow]"
                    )
            console.print(f"  [green]Saved {len(sample_keys)} images to {vis_dir}[/green]")
            if visualize_only:
                continue

        console.print(f"\n[bold]Processing plate [cyan]{barcode}[/cyan]...")
        console.print(f"  Images:   {image_dir}")
        console.print(f"  Features: {feature_dir}")

        extracted, expected, missing = verify_plate_features(feature_dir, image_dir)
        if extracted == expected and not missing:
            console.print(
                f"  [yellow]Skipping[/yellow] existing complete batch ({extracted}/{expected})"
            )
            continue

        if extracted > 0:
            console.print(
                "  [yellow]Incomplete output detected[/yellow] "
                f"({extracted}/{expected}); clearing and re-extracting batch"
            )
            removed_features = _clear_pt_files(feature_dir)
            console.print(f"  Cleared {removed_features} existing feature files")
            if not no_tensors:
                removed_tensors = _clear_pt_files(tensor_dir)
                if removed_tensors:
                    console.print(f"  Cleared {removed_tensors} existing tensor files")

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


# --------------------------------------------------------------------------- #
# pipeline
# --------------------------------------------------------------------------- #


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
    """Autonomous feature extraction pipeline: fetch -> extract -> cleanup.

    Processes plates one at a time, tracking progress for crash-safe resume.
    Designed for unattended overnight runs.
    """
    load_dotenv()

    with open(config) as f:
        cfg = yaml.safe_load(f)["cpjump"]

    if model_name is not None:
        cfg.setdefault("extraction", {})["model"] = model_name
    if features_root is not None:
        cfg.setdefault("local", {})["features"] = str(features_root)
    if tensors_root is not None:
        cfg.setdefault("local", {})["tensors"] = str(tensors_root)

    fetch_cfg = cfg.get("fetch", {})
    backend_name = choose_backend(
        str(backend.value if backend else fetch_cfg.get("backend", "auto"))
    )

    log_path = log_file or Path(f"data/pipeline_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.log")
    setup_pipeline_logging(log_path)

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


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #


@app.command()
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


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #


@app.command()
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
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress_bar:
        overall = progress_bar.add_task("[green]Overall", total=len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    download_and_extract_archive,
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
                    progress_bar.console.print(f"  [green]{result}[/green]")
                except Exception as exc:
                    progress_bar.console.print(f"  [red]{archive}: {exc}[/red]")
                progress_bar.advance(overall)

    console.print("\n[bold green]Download complete.[/bold green]")
    console.print(f"Features extracted to: {output_dir}")
