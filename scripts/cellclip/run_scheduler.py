#!/usr/bin/env python
"""Run or preview the CellCLIP ChemBERTa sweep scheduler."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellclip.scheduler import load_schedule_spec, run_schedule  # noqa: E402

DEFAULT_SPEC = Path("configs/cellclip/schedules/chemberta_full_benchmark.yaml")

console = Console()


def main(
    spec: Annotated[Path, typer.Option(help="Schedule spec YAML.")] = DEFAULT_SPEC,
    dry_run: Annotated[bool, typer.Option(help="Preview the schedule without running it.")] = False,
    resume: Annotated[bool, typer.Option(help="Resume an existing schedule directory.")] = False,
) -> None:
    """Run or preview the CellCLIP ChemBERTa sweep scheduler."""
    schedule_spec = load_schedule_spec(spec)

    console.rule("[bold blue]CellCLIP Scheduler")
    console.print(f"Spec:             {spec}")
    console.print(f"Schedule:         {schedule_spec.schedule_name}")
    console.print(f"Base config:      {schedule_spec.base_config}")
    console.print(f"Dry run:          {dry_run}")
    console.print(f"Resume:           {resume}")

    schedule_dir = run_schedule(schedule_spec, dry_run=dry_run, resume=resume)

    console.rule("[bold green]Scheduler Complete")
    console.print(f"Schedule dir:     {schedule_dir}")
    if not dry_run:
        console.print(f"Manifest:         {schedule_dir / 'manifest.jsonl'}")
        console.print(f"Final report:     {schedule_dir / 'final_report.md'}")


if __name__ == "__main__":
    typer.run(main)
