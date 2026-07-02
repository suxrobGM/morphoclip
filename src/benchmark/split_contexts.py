"""Metadata paths, context dataclasses, and loaders for benchmark splits.

Split-strategy code reads the module-level path constants here *through the
resolvers*, so tests patch ``benchmark.split_contexts.METADATA_PATH`` /
``OFFICIAL_SPLIT_METADATA_PATH`` to redirect them. Keep the constants, resolvers,
and loaders together in this module.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

METADATA_PATH = Path("output/benchmark/input/experiment-metadata.tsv")
FALLBACK_METADATA_PATH = Path("output/benchmark/output/experiment-metadata.tsv")
OFFICIAL_SPLIT_METADATA_PATH = Path("data/reference/cpjump1/cpjump1_metadata.csv")


@dataclass(frozen=True, slots=True)
class BenchmarkPlateContext:
    """Benchmark slice metadata for a single assay plate."""

    cell_type: str
    perturbation: str
    timepoint: int


@dataclass(frozen=True, slots=True)
class OfficialSplitContext:
    """Per-well official CPJUMP1 split metadata."""

    plate: str
    well: str
    broad_sample: str
    target: str
    cell_line: str
    experiment_type: str
    timepoint: int
    timepoint_code: str
    target_is_across: bool
    target_radix: int | None


def resolve_metadata_path() -> Path:
    """Find the experiment metadata TSV."""
    for path in (METADATA_PATH, FALLBACK_METADATA_PATH):
        if path.exists():
            return path
    raise AssertionError(
        f"Experiment metadata not found: {METADATA_PATH} or {FALLBACK_METADATA_PATH}"
    )


def resolve_official_split_metadata_path() -> Path:
    """Find the official CPJUMP1 split CSV."""
    if OFFICIAL_SPLIT_METADATA_PATH.exists():
        return OFFICIAL_SPLIT_METADATA_PATH
    raise AssertionError(
        f"Official CPJUMP1 split metadata not found: {OFFICIAL_SPLIT_METADATA_PATH}"
    )


def load_plate_contexts() -> dict[str, BenchmarkPlateContext]:
    """Load plate-level benchmark metadata from experiment TSV."""
    path = resolve_metadata_path()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Benchmark metadata is empty: {path}")

        required = {"Assay_Plate_Barcode", "Cell_type", "Perturbation", "Time"}
        missing_columns = required - set(reader.fieldnames)
        if missing_columns:
            missing_display = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Missing required columns in benchmark metadata {path}: {missing_display}"
            )

        result: dict[str, BenchmarkPlateContext] = {}
        for row in reader:
            barcode = row.get("Assay_Plate_Barcode", "")
            cell_type = row.get("Cell_type", "")
            perturbation = row.get("Perturbation", "")
            time_str = row.get("Time", "")
            if not (barcode and cell_type and perturbation and time_str):
                continue
            result[barcode] = BenchmarkPlateContext(
                cell_type=cell_type,
                perturbation=perturbation,
                timepoint=int(time_str),
            )

    if not result:
        raise ValueError(
            "No Assay_Plate_Barcode/Cell_type/Perturbation/Time mappings found "
            f"in experiment metadata: {path}"
        )
    return result


def _parse_bool(value: str) -> bool:
    return str(value).strip().upper() == "TRUE"


def _parse_optional_int(value: str) -> int | None:
    stripped = str(value).strip()
    if not stripped or stripped.upper() == "NA":
        return None
    return int(stripped)


def load_official_split_contexts() -> dict[tuple[str, str], OfficialSplitContext]:
    """Load per-well official CPJUMP1 split metadata."""
    path = resolve_official_split_metadata_path()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Official CPJUMP1 split metadata is empty: {path}")

        required = {
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_broad_sample",
            "Metadata_target",
            "Metadata_cell_line",
            "Metadata_experiment_type",
            "Metadata_timepoint",
            "Metadata_timepoint_code",
            "Metadata_target_is_across",
        }
        missing = required - set(reader.fieldnames)
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(
                f"Missing required columns in official split metadata {path}: {missing_display}"
            )

        result: dict[tuple[str, str], OfficialSplitContext] = {}
        for row in reader:
            plate = row.get("Metadata_Plate", "").strip()
            well = row.get("Metadata_Well", "").strip().upper()
            if not plate or not well:
                continue
            result[(plate, well)] = OfficialSplitContext(
                plate=plate,
                well=well,
                broad_sample=row.get("Metadata_broad_sample", "").strip(),
                target=row.get("Metadata_target", "").strip(),
                cell_line=row.get("Metadata_cell_line", "").strip(),
                experiment_type=row.get("Metadata_experiment_type", "").strip(),
                timepoint=int(row.get("Metadata_timepoint", "0")),
                timepoint_code=row.get("Metadata_timepoint_code", "").strip(),
                target_is_across=_parse_bool(row.get("Metadata_target_is_across", "")),
                target_radix=_parse_optional_int(row.get("Metadata_target_radix", "")),
            )

    if not result:
        raise ValueError(f"No official split metadata rows found in {path}")
    return result
