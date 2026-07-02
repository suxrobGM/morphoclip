"""MorphoCLIP training entry point.

Usage (single GPU):
    uv run poe train
    uv run poe train --config configs/train/base.yaml --run-name experiment_1

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 scripts/training/train.py \\
        --config configs/train/ddp.yaml --distributed

Resume:
    uv run poe train --resume output/morphoclip_runs/experiment_1/checkpoints/last.pt
"""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# Route library-level logger.info() calls through rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False, show_path=False)],
)

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoclip.training.config import load_training_config  # noqa: E402
from morphoclip.training.trainer import train_morphoclip  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MorphoCLIP")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train/base.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name (default: from config or 'default')",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to checkpoint to resume from "
            "(e.g. output/morphoclip_runs/default/checkpoints/last.pt)"
        ),
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DDP multi-GPU training (requires torchrun launcher)",
    )
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.distributed:
        config.distributed.enabled = True
    if args.run_name:
        config.runtime.run_name = args.run_name

    run_name = config.runtime.run_name or "default"
    run_dir = Path(config.runtime.output_root) / run_name

    resume_from = Path(args.resume) if args.resume else None
    summary = train_morphoclip(config, run_dir=run_dir, resume_from=resume_from)
    console.print("\n[bold green]Training complete.[/bold green]")
    console.print(f"  Run dir:         {summary['run_dir']}")
    console.print(f"  Train wells:     {summary['train_wells']}")
    console.print(f"  Val wells:       {summary['val_wells']}")
    console.print(f"  Best checkpoint: {summary['best_checkpoint']}")
    console.print(f"  Metrics:         {summary['metrics_path']}")


if __name__ == "__main__":
    main()
