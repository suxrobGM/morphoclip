#!/usr/bin/env python
"""Compare two or more benchmark output directories."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.plot import RunSpec, generate_benchmark_comparison  # noqa: E402

console = Console()


def _parse_run_spec(value: str) -> RunSpec:
    """Parse one ``LABEL=DIR`` run spec."""
    if "=" not in value:
        raise typer.BadParameter(f"Invalid --run value {value!r}; expected LABEL=DIR")
    label, raw_dir = value.split("=", 1)
    label = label.strip()
    raw_dir = raw_dir.strip()
    if not label or not raw_dir:
        raise typer.BadParameter(f"Invalid --run value {value!r}; expected LABEL=DIR")
    run_dir = Path(raw_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    return RunSpec(label=label, run_dir=run_dir)


def main(
    runs: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Named benchmark run in LABEL=DIR form (repeatable), e.g. "
                "--run baseline=output/benchmark --run cellclip=output/benchmark_cellclip_hf."
            )
        ),
    ] = None,
    baseline_dir: Annotated[
        Path, typer.Option(help="Benchmark output directory for the baseline run.")
    ] = PROJECT_ROOT / "output" / "benchmark",
    candidate_dir: Annotated[
        Path, typer.Option(help="Benchmark output directory for the CellCLIP run.")
    ] = PROJECT_ROOT / "output" / "benchmark_cellclip_hf",
    output_dir: Annotated[
        Path, typer.Option(help="Directory where merged tables and plots will be written.")
    ] = PROJECT_ROOT / "output" / "benchmark_comparison",
    baseline_label: Annotated[
        str, typer.Option(help="Legend label for the baseline run.")
    ] = "Baseline",
    candidate_label: Annotated[
        str, typer.Option(help="Legend label for the candidate run.")
    ] = "CellCLIP",
) -> None:
    """Compare two or more benchmark output directories."""
    if runs:
        run_specs = [_parse_run_spec(value) for value in runs]
    else:
        run_specs = [
            RunSpec(label=baseline_label, run_dir=baseline_dir),
            RunSpec(label=candidate_label, run_dir=candidate_dir),
        ]

    _, table_paths, plot_paths = generate_benchmark_comparison(
        run_specs=run_specs,
        output_dir=output_dir,
    )

    console.print("Saved comparison tables:")
    for path in table_paths.values():
        console.print(f"  {path}")

    console.print("Saved comparison plots:")
    for path in plot_paths.values():
        console.print(f"  {path}")


if __name__ == "__main__":
    typer.run(main)
