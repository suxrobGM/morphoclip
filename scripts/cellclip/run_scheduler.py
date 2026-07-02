#!/usr/bin/env python
"""Run or preview the CellCLIP ChemBERTa sweep scheduler."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellclip.scheduler import load_schedule_spec, run_schedule  # noqa: E402

DEFAULT_SPEC = Path("configs/cellclip/schedules/chemberta_full_benchmark.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the scheduler CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = load_schedule_spec(args.spec)

    print("=" * 60)
    print("CellCLIP Scheduler")
    print("=" * 60)
    print(f"Spec:             {args.spec}")
    print(f"Schedule:         {spec.schedule_name}")
    print(f"Base config:      {spec.base_config}")
    print(f"Dry run:          {args.dry_run}")
    print(f"Resume:           {args.resume}")
    print("=" * 60)

    schedule_dir = run_schedule(spec, dry_run=args.dry_run, resume=args.resume)

    print("=" * 60)
    print("Scheduler Complete")
    print("=" * 60)
    print(f"Schedule dir:     {schedule_dir}")
    if not args.dry_run:
        print(f"Manifest:         {schedule_dir / 'manifest.jsonl'}")
        print(f"Final report:     {schedule_dir / 'final_report.md'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
