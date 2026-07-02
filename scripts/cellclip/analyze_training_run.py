#!/usr/bin/env python
"""Analyze a local CellCLIP training run and write reusable diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellclip.training.analysis import (  # noqa: E402
    build_comparison,
    build_run_summary,
    write_analysis_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the analysis CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compare-run-dir", type=Path, default=None)
    parser.add_argument("--compare-benchmark-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-eval-wells", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or (run_dir / "analysis")).resolve()

    primary = build_run_summary(
        run_dir,
        max_eval_wells=args.max_eval_wells,
        compare_benchmark_dir=args.compare_benchmark_dir,
    )
    secondary = None
    comparison = None
    if args.compare_run_dir is not None:
        secondary = build_run_summary(
            args.compare_run_dir.resolve(),
            max_eval_wells=args.max_eval_wells,
        )
        comparison = build_comparison(primary, secondary)

    summary_path, report_path = write_analysis_outputs(
        output_dir,
        primary,
        secondary=secondary,
        comparison=comparison,
    )
    print("=" * 60)
    print("CellCLIP Analysis Complete")
    print("=" * 60)
    print(f"Run directory:     {run_dir}")
    print(f"Summary JSON:      {summary_path}")
    print(f"Report Markdown:   {report_path}")
    if args.compare_run_dir is not None:
        print(f"Compare run:       {args.compare_run_dir.resolve()}")
    if args.compare_benchmark_dir is not None:
        print(f"Benchmark tables:  {args.compare_benchmark_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
