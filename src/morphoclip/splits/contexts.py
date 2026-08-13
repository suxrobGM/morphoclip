"""Metadata paths, context dataclasses, and loaders for benchmark splits.

Split-strategy code reads the module-level path constants here *through the
resolvers*, so tests patch ``morphoclip.splits.contexts.METADATA_PATH`` /
``OFFICIAL_SPLIT_METADATA_PATH`` to redirect them. Keep the constants, resolvers,
and loaders together in this module.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

METADATA_PATH = Path("output/benchmark/input/experiment-metadata.tsv")
FALLBACK_METADATA_PATH = Path("output/benchmark/output/experiment-metadata.tsv")
REFERENCE_METADATA_PATH = Path("data/reference/cpjump1/experiment-metadata.tsv")
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
    """Find the experiment metadata TSV.

    Regenerated copies under ``output/benchmark/`` win over the vendored
    reference copy, so a rerun of the upstream notebook takes effect without
    touching the repo.
    """
    path = _resolve_optional_metadata_path()
    if path is None:
        raise AssertionError(
            f"Experiment metadata not found: {METADATA_PATH}, {FALLBACK_METADATA_PATH}, "
            f"or {REFERENCE_METADATA_PATH}"
        )
    return path


def _resolve_optional_metadata_path() -> Path | None:
    for path in (METADATA_PATH, FALLBACK_METADATA_PATH, REFERENCE_METADATA_PATH):
        if path.exists():
            return path
    return None


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


def load_plate_conditions() -> dict[str, str]:
    """Map each plate barcode to its experimental condition key.

    The condition is ``"<cell_line>|<experiment_type>|<timepoint>"``, which is
    what makes two plates replicates of each other. Cross-Well Alignment groups
    plates by this key so it removes replicate-to-replicate drift without
    deleting the condition-level signal the text prompts encode.

    Plates absent from the official CSV (the non-benchmark-eligible conditions:
    non-standard seeding density, antibiotics present, compound plates on the
    Cas9 line) fall back to the experiment metadata TSV, keyed on every
    condition axis it records. The two sources deliberately produce different
    key shapes: replicate groups never straddle the sources, and distinct
    shapes cannot collide into one group by accident.

    Returns:
        Plate barcode to condition key.

    Raises:
        ValueError: If two wells of the same plate disagree on the condition.
    """
    conditions: dict[str, str] = {}
    for context in load_official_split_contexts().values():
        key = f"{context.cell_line}|{context.experiment_type}|{context.timepoint}"
        existing = conditions.setdefault(context.plate, key)
        if existing != key:
            raise ValueError(
                f"Plate {context.plate!r} has conflicting conditions in the official "
                f"split metadata: {existing!r} and {key!r}"
            )
    for barcode, key in _load_experiment_conditions().items():
        conditions.setdefault(barcode, key)
    return conditions


_EXPERIMENT_CONDITION_FIELDS = (
    "Cell_type",
    "Perturbation",
    "Time",
    "Density",
    "Antibiotics",
    "Cell_line",
    "Time_delay",
)


def _load_experiment_conditions() -> dict[str, str]:
    """Condition keys from the experiment TSV, for plates the official CSV lacks.

    Density, Antibiotics, Cell_line (Parental vs Cas9), and Time_delay must be
    part of the key: they are exactly what separates the non-benchmark plates
    from the standard-condition plates that otherwise share cell type,
    modality, and timepoint.

    Raises:
        ValueError: If the TSV exists but lacks a column the key needs. Skipping
            the check would return an empty map, which reaches training as a
            zero CWA offset on every plate the official CSV does not cover.
    """
    path = _resolve_optional_metadata_path()
    if path is None:
        return {}

    result: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Experiment metadata is empty: {path}")

        required = {"Assay_Plate_Barcode", *_EXPERIMENT_CONDITION_FIELDS}
        missing_columns = required - set(reader.fieldnames)
        if missing_columns:
            missing_display = ", ".join(sorted(missing_columns))
            raise ValueError(
                f"Missing required columns in experiment metadata {path}: {missing_display}"
            )

        for row in reader:
            barcode = row.get("Assay_Plate_Barcode", "")
            values = [row.get(field, "") for field in _EXPERIMENT_CONDITION_FIELDS]
            if not barcode or not all(values):
                continue
            result[barcode] = "|".join(values)
    return result
