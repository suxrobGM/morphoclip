"""CSV / JSON / manifest I/O for the CellCLIP sweep scheduler."""

import csv
import json
from pathlib import Path
from typing import Any


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the latest manifest row for each stage/candidate pair."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["stage"], row["candidate_id"])
            records[key] = row
    return records


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    """Append one scheduler record to a JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
