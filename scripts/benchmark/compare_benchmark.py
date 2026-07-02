#!/usr/bin/env python
"""Compare two or more benchmark output directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.plot import RunSpec, generate_benchmark_comparison  # noqa: E402


def _parse_run_spec(value: str) -> RunSpec:
    """Parse one ``LABEL=DIR`` run spec."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Invalid --run value {value!r}; expected LABEL=DIR")
    label, raw_dir = value.split("=", 1)
    label = label.strip()
    raw_dir = raw_dir.strip()
    if not label or not raw_dir:
        raise argparse.ArgumentTypeError(f"Invalid --run value {value!r}; expected LABEL=DIR")
    run_dir = Path(raw_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    return RunSpec(label=label, run_dir=run_dir)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compare two or more benchmark output directories."
    )
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        type=_parse_run_spec,
        default=None,
        help=(
            "Named benchmark run in LABEL=DIR form. Repeat for multiple runs, "
            "for example --run baseline=output/benchmark "
            "--run cellclip=output/benchmark_cellclip_hf."
        ),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "benchmark",
        help="Benchmark output directory for the baseline run.",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "benchmark_cellclip_hf",
        help="Benchmark output directory for the CellCLIP run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "benchmark_comparison",
        help="Directory where merged tables and plots will be written.",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="Baseline",
        help="Legend label for the baseline run.",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="CellCLIP",
        help="Legend label for the candidate run.",
    )
    return parser


def main() -> None:
    """Run comparison plotting."""
    args = build_parser().parse_args()
    run_specs = args.runs
    if not run_specs:
        run_specs = [
            RunSpec(label=args.baseline_label, run_dir=args.baseline_dir),
            RunSpec(label=args.candidate_label, run_dir=args.candidate_dir),
        ]

    _, table_paths, plot_paths = generate_benchmark_comparison(
        run_specs=run_specs,
        output_dir=args.output_dir,
    )

    print("Saved comparison tables:")
    for path in table_paths.values():
        print(f"  {path}")

    print("Saved comparison plots:")
    for path in plot_paths.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
