"""Benchmark reporting helpers for the CellCLIP scheduler."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from cellclip.scheduler_spec import BENCHMARK_COMPARE_KEYS

if TYPE_CHECKING:
    from cellclip.scheduler_spec import ScheduleSpec


def _read_summary_table(path: Path, kind: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    normalized: list[dict[str, str]] = []
    if kind in {"replicability", "matching"}:
        for row in rows:
            for cell in ("A549", "U2OS"):
                value = row.get(cell, "")
                if value:
                    normalized.append(
                        {"key": f"{row['Modality']}|{cell}|{row['time']}", "candidate": value}
                    )
    else:
        for row in rows:
            for cell in ("A549", "U2OS"):
                value = row.get(cell, "")
                if value:
                    normalized.append(
                        {
                            "key": f"{row['Modality1']}->{row['Modality2']}|{cell}",
                            "candidate": value,
                        }
                    )
    return normalized


def _comparison_lookup(benchmark_dir: Path, kind: str) -> dict[str, str]:
    filename = {
        "replicability": "replicability_summary.csv",
        "matching": "matching_summary.csv",
        "gene_compound": "gene_compound_matching_summary.csv",
    }[kind]
    return {
        row["key"]: row["candidate"]
        for row in _read_summary_table(benchmark_dir / "tables" / filename, kind)
    }


def write_final_report(
    spec: ScheduleSpec, schedule_dir: Path, records: list[dict[str, object]]
) -> Path:
    """Write a Markdown report for final-stage full benchmarks."""
    report_path = schedule_dir / "final_report.md"
    timeline_label = ", ".join(spec.benchmark_timelines)
    lines = [
        f"# {spec.schedule_name}",
        "",
        f"Final-stage benchmark comparison for timelines: {timeline_label}.",
    ]
    for record in records:
        if record.get("status") != "completed" or "benchmark_dir" not in record:
            continue
        benchmark_dir = Path(str(record["benchmark_dir"]))
        lines.extend(
            [
                "",
                f"## {record['candidate_id']}",
                "",
                f"- Run dir: `{record['run_dir']}`",
                f"- Benchmark dir: `{benchmark_dir}`",
            ]
        )
        for title, kind in (
            ("Replicability", "replicability"),
            ("Matching", "matching"),
            ("Gene-Compound Matching", "gene_compound"),
        ):
            tables = {
                label: _comparison_lookup(spec.compare_full_benchmark_dirs[label], kind)
                for label in BENCHMARK_COMPARE_KEYS
            }
            candidate = _comparison_lookup(benchmark_dir, kind)
            keys = sorted(set(candidate) | set(tables["baseline"]) | set(tables["pretrained_clip"]))
            lines.extend(
                [
                    "",
                    f"### {title}",
                    "",
                    "| Key | Candidate | Baseline | Pretrained Clip |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for key in keys:
                baseline = tables["baseline"].get(key, "-")
                pretrained = tables["pretrained_clip"].get(key, "-")
                lines.append(f"| {key} | {candidate.get(key, '-')} | {baseline} | {pretrained} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
