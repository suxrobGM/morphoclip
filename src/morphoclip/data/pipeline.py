"""Autonomous plate extraction pipeline with progress tracking and resume.

Orchestrates the full fetch -> extract -> cleanup cycle for all plates.
Designed for unattended overnight runs with automatic resume on interruption.

Usage via CLI::

    uv run poe extract-pipeline
    uv run poe extract-pipeline --retry-failed
    uv run poe extract-pipeline --tensors-only

CLI entry point: ``scripts/features/run_pipeline.py``.
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from morphoclip.data.feature_extractor import (
    extract_plate_features_with_model,
    feature_filename,
    load_dinov3,
    verify_plate_features,
)
from morphoclip.data.image_loader import (
    DINO_INPUT_SIZE,
    FLUORESCENCE_CHANNELS,
    discover_sites,
    load_site_as_tensor,
)
from morphoclip.data.perturbation import extract_plate_barcode
from morphoclip.data.progress import (
    _COMPLETENESS_THRESHOLD,
    _MAX_SITES_PER_PLATE,
    PlateRecord,
    PlateStatus,
    _utcnow,
    compute_config_hash,
    load_progress,
    save_progress,
)
from morphoclip.utils.s3 import build_s3_uri, sync_s3_path

logger = logging.getLogger(__name__)
console = Console()


def _count_tiffs(image_dir: Path) -> int:
    """Count TIFF files in a directory."""
    return len(list(image_dir.glob("*.tif"))) + len(list(image_dir.glob("*.tiff")))


def _delete_tiffs(image_dir: Path, *, dry_run: bool = False) -> int:
    """Delete original TIFF images. Returns deleted file count."""
    tif_paths = sorted(image_dir.glob("*.tif")) + sorted(image_dir.glob("*.tiff"))
    if not dry_run:
        for p in tif_paths:
            p.unlink(missing_ok=True)
    return len(tif_paths)


def setup_logging(log_path: Path) -> None:
    """Configure dual logging: file (DEBUG) + console (INFO).

    Args:
        log_path: Path to the log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)


class PlateExtractionPipeline:
    """Autonomous fetch -> extract -> cleanup pipeline for all plates.

    Processes plates one at a time to minimize disk usage. Tracks progress
    in a JSON file for crash-safe resume.

    Args:
        config: The ``cpjump`` config dict from ``configs/dataset.yml``.
        progress_path: Path to the progress JSON file.
        backend: S3 transfer backend (``"awscli"`` or ``"rclone"``).
        save_tensors: Whether to save resized image tensors alongside features.
        tensors_only: Save only resized tensors, skip DINOv3 extraction.
        dry_run: Log all steps without executing S3 sync or deletion.
        retry_failed: Reset failed plates to pending before starting.
    """

    def __init__(
        self,
        *,
        config: dict[str, Any],
        progress_path: Path,
        backend: str,
        save_tensors: bool = False,
        tensors_only: bool = False,
        dry_run: bool = False,
        retry_failed: bool = False,
    ) -> None:
        self._config = config
        self._progress_path = progress_path
        self._backend = backend
        self._save_tensors = save_tensors or tensors_only
        self._tensors_only = tensors_only
        self._dry_run = dry_run
        self._retry_failed = retry_failed

        local = config.get("local", {})
        self._raw_root = Path(local.get("raw_images", "data/raw"))
        self._features_root = Path(local.get("features", "data/features"))
        self._tensors_root = Path(local.get("tensors", "data/tensors"))
        self._metadata_root = Path(local.get("metadata", "data/metadata"))

        self._batch = config.get("batch", "")
        self._endpoint = config["endpoint"]
        self._plates: list[str] = config.get("plates", [])

        fetch_cfg = config.get("fetch", {})
        self._no_sign_request = bool(fetch_cfg.get("aws_no_sign_request", True))
        self._rclone_remote = str(
            fetch_cfg.get(
                "rclone_remote", ":s3,provider=AWS,region=us-east-1,no_check_bucket=true:"
            )
        )

        extraction = config.get("extraction", {})
        self._model_name = extraction.get("model", "facebook/dinov3-vitl16-pretrain-lvd1689m")
        self._device = extraction.get("device", "auto")
        self._batch_size = extraction.get("batch_size", 48)

        self._progress = load_progress(progress_path, compute_config_hash(self._plates))

    def run(
        self,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        plates: list[str] | None = None,
    ) -> None:
        """Execute the full pipeline: metadata -> plates loop.

        Args:
            device: Override device from config.
            batch_size: Override batch size from config.
            plates: Restrict to specific plate names or barcodes.
        """
        device = device or self._device
        batch_size = batch_size or self._batch_size
        plate_list = self._resolve_plates(plates)

        console.rule("[bold blue]Autonomous Feature Extraction Pipeline")
        console.print(f"  Plates:      {len(plate_list)}/{len(self._plates)}")
        console.print(f"  Backend:     {self._backend}")
        console.print(f"  Device:      {device}")
        console.print(f"  Batch size:  {batch_size}")
        tensor_mode = "only" if self._tensors_only else "yes" if self._save_tensors else "no"
        console.print(f"  Tensors:     {tensor_mode}")
        console.print(f"  Dry run:     {self._dry_run}")
        console.print(f"  Progress:    {self._progress_path}")

        if self._retry_failed:
            reset_count = self._reset_failed_plates(plate_list)
            if reset_count:
                console.print(f"  [yellow]Reset {reset_count} failed plates to pending[/yellow]")

        model = None
        processor = None
        if not self._tensors_only:
            console.print(f"\n[bold]Loading vision model: {self._model_name}[/bold]")
            model, processor = load_dinov3(self._model_name, device)
            console.print(f"  [green]Model loaded on {device}[/green]")

        self._fetch_metadata()

        completed = 0
        skipped = 0
        failed = 0

        for i, plate_name in enumerate(plate_list, 1):
            barcode = extract_plate_barcode(plate_name)
            console.rule(f"[bold cyan]Plate {i}/{len(plate_list)}: {barcode}")

            record = self._get_record(plate_name)
            record["barcode"] = barcode

            if record["status"] in {PlateStatus.COMPLETED, PlateStatus.SKIPPED}:
                console.print(f"  [dim]Already {record['status']}, skipping[/dim]")
                skipped += 1
                continue

            feature_dir = self._features_root / barcode
            tensor_dir = self._tensors_root / barcode
            if self._check_output_complete(plate_name, feature_dir, tensor_dir):
                record["status"] = PlateStatus.SKIPPED
                record["completed_at"] = _utcnow()
                self._save_progress()
                console.print("  [green]Output already complete, skipped[/green]")
                skipped += 1
                continue

            image_dir = self._image_dir(plate_name)
            try:
                record["status"] = PlateStatus.FETCHING
                record["started_at"] = _utcnow()
                record["error"] = None
                self._save_progress()

                self._fetch_plate(plate_name, image_dir)

                record["status"] = PlateStatus.EXTRACTING
                self._save_progress()

                if self._tensors_only:
                    sites_count = self._save_tensors_only(image_dir, tensor_dir, batch_size)
                else:
                    assert model is not None and processor is not None
                    saved = extract_plate_features_with_model(
                        image_dir,
                        feature_dir,
                        model,
                        processor,
                        device=device,
                        batch_size=batch_size,
                        save_tensors=self._save_tensors,
                        tensor_output_dir=tensor_dir if self._save_tensors else None,
                    )
                    sites_count = len(saved)

                record["status"] = PlateStatus.COMPLETED
                record["sites_extracted"] = sites_count
                record["completed_at"] = _utcnow()
                self._save_progress()

                console.print(f"  [green]Completed: {sites_count} sites[/green]")
                completed += 1

            except Exception as exc:
                record["status"] = PlateStatus.FAILED
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["completed_at"] = _utcnow()
                self._save_progress()
                logger.error("Plate %s failed: %s", plate_name, exc, exc_info=True)
                console.print(f"  [bold red]Failed: {exc}[/bold red]")
                failed += 1

            finally:
                self._cleanup_plate(image_dir)

        console.rule("[bold blue]Pipeline Complete")
        console.print(f"  Completed: {completed}")
        console.print(f"  Skipped:   {skipped}")
        console.print(f"  Failed:    {failed}")
        console.print(f"  Progress:  {self._progress_path}")

    def _save_progress(self) -> None:
        """Persist current progress to disk."""
        save_progress(self._progress, self._progress_path)

    def _image_dir(self, plate_name: str) -> Path:
        """Resolve the raw ``Images/`` directory for a plate."""
        return self._raw_root / self._batch / plate_name / "Images"

    def _sync(self, uri: str, dest: Path) -> None:
        """Sync an S3 path into *dest* using the pipeline's backend settings."""
        sync_s3_path(
            uri,
            dest,
            backend=self._backend,
            no_sign_request=self._no_sign_request,
            rclone_remote=self._rclone_remote,
            dry_run=self._dry_run,
        )

    def _compute_config_hash(self) -> str:
        """SHA-256 of the sorted plate list."""
        return compute_config_hash(self._plates)

    def _get_record(self, plate_name: str) -> dict[str, Any]:
        """Get or create a plate record in the progress dict."""
        if plate_name not in self._progress.plates:
            record = asdict(PlateRecord(barcode=extract_plate_barcode(plate_name)))
            self._progress.plates[plate_name] = record
        return self._progress.plates[plate_name]

    def _reset_failed_plates(self, plate_list: list[str]) -> int:
        """Reset all FAILED plates in *plate_list* to PENDING.

        Returns:
            Number of plates reset.
        """
        reset_count = 0
        for plate_name in plate_list:
            record = self._get_record(plate_name)
            if record["status"] == PlateStatus.FAILED:
                record["status"] = PlateStatus.PENDING
                record["error"] = None
                reset_count += 1
        if reset_count:
            self._save_progress()
        return reset_count

    def _resolve_plates(self, plates: list[str] | None) -> list[str]:
        """Resolve plate filter to full plate names from config."""
        if not plates:
            return list(self._plates)

        resolved = []
        for p in plates:
            matches = [
                name for name in self._plates if name == p or extract_plate_barcode(name) == p
            ]
            if matches:
                resolved.extend(matches)
            else:
                logger.warning("Plate %s not found in config, skipping", p)
        return resolved

    def _check_output_complete(
        self,
        plate_name: str,
        feature_dir: Path,
        tensor_dir: Path,
    ) -> bool:
        """Check if output already exists without needing raw images.

        Uses the progress record if available, falls back to filesystem
        heuristics.
        """
        check_dir = tensor_dir if self._tensors_only else feature_dir

        if not check_dir.exists():
            return False

        actual = len(list(check_dir.glob("*.pt")))
        if actual == 0:
            return False

        min_sites = int(_MAX_SITES_PER_PLATE * _COMPLETENESS_THRESHOLD)

        # Enforce min_sites alongside recorded count to catch stale partial records
        record = self._progress.plates.get(plate_name)
        if record and record["status"] == PlateStatus.COMPLETED and record["sites_extracted"] > 0:
            return actual >= max(record["sites_extracted"], min_sites)

        image_dir = self._image_dir(plate_name)
        if not self._tensors_only and image_dir.exists():
            _, _, missing = verify_plate_features(feature_dir, image_dir)
            return len(missing) == 0

        return actual >= min_sites

    def _fetch_metadata(self) -> None:
        """Download platemap + external metadata if not already present."""
        if self._progress.metadata_downloaded:
            return

        console.print("\n[bold]Downloading metadata...[/bold]")
        platemaps_uri = build_s3_uri(self._endpoint, self._config["metadata"], self._batch)
        self._sync(platemaps_uri, self._metadata_root / "platemaps" / self._batch)

        ext_metadata_uri = build_s3_uri(
            self._endpoint, self._config["external_metadata"], self._batch
        )
        self._sync(ext_metadata_uri, self._metadata_root / "external_metadata")

        self._progress.metadata_downloaded = True
        self._save_progress()
        console.print("  [green]Metadata downloaded[/green]")

    def _fetch_plate(self, plate_name: str, image_dir: Path) -> None:
        """Download plate images from S3."""
        existing_tiffs = _count_tiffs(image_dir) if image_dir.exists() else 0
        if existing_tiffs > 0:
            console.print(f"  Raw images already present ({existing_tiffs} TIFFs)")
            return

        images_uri = build_s3_uri(self._endpoint, self._config["images"], self._batch)
        plate_uri = f"{images_uri}/{plate_name}/Images"

        console.print("  Fetching from S3...")
        self._sync(plate_uri, image_dir)

        tiff_count = _count_tiffs(image_dir)
        console.print(f"  Downloaded {tiff_count} TIFFs")
        logger.info("Fetched plate %s: %d TIFFs", plate_name, tiff_count)

    def _save_tensors_only(
        self,
        image_dir: Path,
        tensor_dir: Path,
        batch_size: int,
    ) -> int:
        """Save only resized image tensors, no DINOv3 extraction.

        Args:
            image_dir: Path to the plate's ``Images/`` directory.
            tensor_dir: Directory to save tensor ``.pt`` files.
            batch_size: Number of sites to process per batch (for progress display).

        Returns:
            Number of sites saved.
        """
        sites = discover_sites(image_dir, channels=FLUORESCENCE_CHANNELS)
        if not sites:
            logger.warning("No complete sites found in %s", image_dir)
            return 0

        tensor_dir.mkdir(parents=True, exist_ok=True)
        site_keys = sorted(sites.keys(), key=str)
        saved = 0

        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Saving tensors", total=len(site_keys))

            for key in site_keys:
                site_tensor = load_site_as_tensor(sites[key], resize=DINO_INPUT_SIZE)
                tensor_path = tensor_dir / feature_filename(key)
                torch.save(site_tensor, tensor_path)
                saved += 1
                progress.advance(task)

        logger.info("Saved %d tensors -> %s", saved, tensor_dir)
        return saved

    def _cleanup_plate(self, image_dir: Path) -> None:
        """Delete raw TIFFs to free disk space.

        Catches all exceptions internally so that cleanup errors in a
        ``finally`` block never mask the original exception.
        """
        try:
            if not image_dir.exists():
                return

            tiff_count = _count_tiffs(image_dir)
            if tiff_count == 0:
                return

            deleted = _delete_tiffs(image_dir, dry_run=self._dry_run)
            if deleted:
                console.print(f"  Cleaned up {deleted} TIFFs")
                logger.info("Cleanup: deleted %d TIFFs from %s", deleted, image_dir)

            if not self._dry_run:
                if image_dir.exists() and not any(image_dir.iterdir()):
                    image_dir.rmdir()
                plate_dir = image_dir.parent
                if plate_dir.exists() and not any(plate_dir.iterdir()):
                    plate_dir.rmdir()

        except Exception as exc:
            logger.warning("Cleanup failed for %s: %s", image_dir, exc)
