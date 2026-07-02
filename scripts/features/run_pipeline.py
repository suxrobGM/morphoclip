"""Autonomous feature extraction pipeline: fetch -> extract -> cleanup.

Processes plates one at a time, tracking progress in a JSON file
for crash-safe resume. Designed for unattended overnight runs.

Usage:
    uv run poe extract-pipeline
    uv run poe extract-pipeline --retry-failed
    uv run poe extract-pipeline --save-tensors
    uv run poe extract-pipeline --tensors-only
    uv run poe extract-pipeline --plates BR00116991 BR00116992
    uv run poe extract-pipeline --dry-run
"""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morphoclip.data.pipeline import PlateExtractionPipeline, setup_logging  # noqa: E402
from morphoclip.utils.s3 import choose_backend  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous feature extraction pipeline: fetch -> extract -> cleanup.",
    )
    p.add_argument("--config", type=Path, default=Path("configs/dataset.yml"))
    p.add_argument(
        "--progress",
        type=Path,
        default=Path("data/pipeline_progress.json"),
        help="Progress file path (default: data/pipeline_progress.json).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Log file path (default: data/pipeline_{timestamp}.log).",
    )
    p.add_argument("--backend", choices=["auto", "awscli", "rclone"], default=None)
    p.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override the vision backbone model ID used for extraction.",
    )
    p.add_argument(
        "--features-root",
        type=Path,
        default=None,
        help="Override the output root for extracted feature .pt files.",
    )
    p.add_argument(
        "--tensors-root",
        type=Path,
        default=None,
        help="Override the output root for saved resized tensors.",
    )
    p.add_argument("--device", type=str, default=None, help="Override device (e.g. cuda, cpu).")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument(
        "--save-tensors",
        action="store_true",
        help="Also save resized (5,384,384) image tensors alongside features.",
    )
    p.add_argument(
        "--tensors-only",
        action="store_true",
        help="Save only resized tensors, skip DINOv3 feature extraction (no GPU needed).",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset failed plates to pending and retry them.",
    )
    p.add_argument(
        "--plates",
        nargs="+",
        type=str,
        default=None,
        help="Restrict to specific plate names or barcodes.",
    )
    p.add_argument("--dry-run", action="store_true", help="Log all steps without executing.")
    return p


def main() -> None:
    args = build_parser().parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)["cpjump"]

    if args.model_name is not None:
        config.setdefault("extraction", {})["model"] = args.model_name
    if args.features_root is not None:
        config.setdefault("local", {})["features"] = str(args.features_root)
    if args.tensors_root is not None:
        config.setdefault("local", {})["tensors"] = str(args.tensors_root)

    # Resolve backend
    fetch_cfg = config.get("fetch", {})
    backend = choose_backend(str(args.backend or fetch_cfg.get("backend", "auto")))

    # Setup logging
    log_path = args.log_file or Path(
        f"data/pipeline_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}.log"
    )
    setup_logging(log_path)

    pipeline = PlateExtractionPipeline(
        config=config,
        progress_path=args.progress,
        backend=backend,
        save_tensors=args.save_tensors,
        tensors_only=args.tensors_only,
        dry_run=args.dry_run,
        retry_failed=args.retry_failed,
    )
    pipeline.run(
        device=args.device,
        batch_size=args.batch_size,
        plates=args.plates,
    )


if __name__ == "__main__":
    main()
