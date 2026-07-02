"""`morphoclip train` command."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from morphoclip.cli.logging import setup_logging
from morphoclip.training.config import load_training_config
from morphoclip.training.trainer import train_morphoclip

console = Console()


def train(
    config: Annotated[
        Path,
        typer.Option(help="Path to training config YAML."),
    ] = Path("configs/train/base.yaml"),
    run_name: Annotated[
        str | None,
        typer.Option(help="Override run name (default: from config or 'default')."),
    ] = None,
    resume: Annotated[
        Path | None,
        typer.Option(
            help="Checkpoint to resume from "
            "(e.g. output/morphoclip_runs/default/checkpoints/last.pt).",
        ),
    ] = None,
    distributed: Annotated[
        bool,
        typer.Option(help="Enable DDP multi-GPU training (requires torchrun launcher)."),
    ] = False,
) -> None:
    """Train MorphoCLIP.

    Single GPU:  morphoclip train --config configs/train/base.yaml --run-name exp_1

    Multi-GPU:   torchrun --nproc_per_node=4 -m morphoclip.cli train \\
                     --config configs/train/ddp.yaml --distributed
    """
    setup_logging()

    training_config = load_training_config(str(config))
    if distributed:
        training_config.distributed.enabled = True
    if run_name:
        training_config.runtime.run_name = run_name

    resolved_run_name = training_config.runtime.run_name or "default"
    run_dir = Path(training_config.runtime.output_root) / resolved_run_name

    summary = train_morphoclip(training_config, run_dir=run_dir, resume_from=resume)
    console.print("\n[bold green]Training complete.[/bold green]")
    console.print(f"  Run dir:         {summary['run_dir']}")
    console.print(f"  Train wells:     {summary['train_wells']}")
    console.print(f"  Val wells:       {summary['val_wells']}")
    console.print(f"  Best checkpoint: {summary['best_checkpoint']}")
    console.print(f"  Metrics:         {summary['metrics_path']}")
