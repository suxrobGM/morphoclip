#!/usr/bin/env python
"""Analyze a local CellCLIP training run and write reusable diagnostics."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from cellclip.training.analysis import (
    build_comparison,
    build_run_summary,
    write_analysis_outputs,
)

console = Console()


def main(
    run_dir: Annotated[Path, typer.Option(help="Training run directory to analyze.")],
    compare_run_dir: Annotated[
        Path | None, typer.Option(help="Second run directory to compare against.")
    ] = None,
    compare_benchmark_dir: Annotated[
        Path | None, typer.Option(help="Benchmark tables directory for comparison.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Output directory (default: <run-dir>/analysis).")
    ] = None,
    max_eval_wells: Annotated[
        int | None, typer.Option(help="Cap the number of eval wells analyzed.")
    ] = None,
) -> None:
    """Analyze a local CellCLIP training run and write reusable diagnostics."""
    resolved_run_dir = run_dir.resolve()
    resolved_output_dir = (output_dir or (resolved_run_dir / "analysis")).resolve()

    primary = build_run_summary(
        resolved_run_dir,
        max_eval_wells=max_eval_wells,
        compare_benchmark_dir=compare_benchmark_dir,
    )
    secondary = None
    comparison = None
    if compare_run_dir is not None:
        secondary = build_run_summary(
            compare_run_dir.resolve(),
            max_eval_wells=max_eval_wells,
        )
        comparison = build_comparison(primary, secondary)

    summary_path, report_path = write_analysis_outputs(
        resolved_output_dir,
        primary,
        secondary=secondary,
        comparison=comparison,
    )
    console.rule("[bold green]CellCLIP Analysis Complete")
    console.print(f"Run directory:     {resolved_run_dir}")
    console.print(f"Summary JSON:      {summary_path}")
    console.print(f"Report Markdown:   {report_path}")
    if compare_run_dir is not None:
        console.print(f"Compare run:       {compare_run_dir.resolve()}")
    if compare_benchmark_dir is not None:
        console.print(f"Benchmark tables:  {compare_benchmark_dir.resolve()}")


if __name__ == "__main__":
    typer.run(main)
