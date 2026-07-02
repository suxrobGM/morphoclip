"""Benchmark-aligned split strategies requiring external metadata.

These strategies depend on metadata from the Chandrasekaran 2024 baselines
(``baselines/2024_Chandrasekaran_NatureMethods_CPJUMP1/datasplits/``) or
the benchmark experiment metadata TSV.

Strategies:
    - ``cpjump1_official_representation``
    - ``cpjump1_official_gene_compound``
    - ``cellclip_cpjump_style``
"""

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from torch.utils.data import Subset

from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.perturbation import (
    PerturbationInfo,
    extract_plate_barcode,
    is_control_or_empty,
)

logger = logging.getLogger(__name__)

METADATA_PATH = Path("output/benchmark/input/experiment-metadata.tsv")
FALLBACK_METADATA_PATH = Path("output/benchmark/output/experiment-metadata.tsv")
OFFICIAL_SPLIT_METADATA_PATH = Path(
    "baselines/2024_Chandrasekaran_NatureMethods_CPJUMP1/datasplits/cpjump1_metadata.csv"
)


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


# --- Shared iteration helpers ---


def _iterate_official_wells(
    dataset: MorphoCLIPDataset,
    contexts: dict[tuple[str, str], OfficialSplitContext],
) -> tuple[list[tuple[int, OfficialSplitContext, PerturbationInfo]], set[tuple[str, str]]]:
    """Iterate dataset wells against official split contexts, filtering controls.

    Returns (annotated_wells, missing_wells).
    """
    annotated: list[tuple[int, OfficialSplitContext, PerturbationInfo]] = []
    missing: set[tuple[str, str]] = set()

    for i, (plate, well, _) in enumerate(dataset.index_entries):
        barcode = extract_plate_barcode(plate)
        context = contexts.get((barcode, well))
        if context is None:
            missing.add((barcode, well))
            continue
        info = dataset.metadata.lookup(barcode, well)
        if is_control_or_empty(info):
            continue
        annotated.append((i, context, info))

    return annotated, missing


def _raise_for_missing_wells(missing: set[tuple[str, str]], path: Path) -> None:
    """Raise ValueError if any wells are missing from the context."""
    if missing:
        preview = ", ".join(f"{plate}/{well}" for plate, well in sorted(missing)[:5])
        raise ValueError(
            f"Missing official split metadata for {len(missing)} well(s) in {path}: {preview}"
        )


def _iterate_plate_wells(
    dataset: MorphoCLIPDataset,
    plate_contexts: dict[str, BenchmarkPlateContext],
) -> tuple[list[tuple[int, BenchmarkPlateContext, PerturbationInfo]], set[str]]:
    """Iterate dataset wells against plate-level contexts, filtering controls.

    Returns (annotated_wells, missing_barcodes).
    """
    annotated: list[tuple[int, BenchmarkPlateContext, PerturbationInfo]] = []
    missing: set[str] = set()

    for i, (plate, well, _) in enumerate(dataset.index_entries):
        barcode = extract_plate_barcode(plate)
        context = plate_contexts.get(barcode)
        if context is None:
            missing.add(barcode)
            continue
        info = dataset.metadata.lookup(barcode, well)
        if is_control_or_empty(info):
            continue
        annotated.append((i, context, info))

    return annotated, missing


def _raise_for_missing_barcodes(missing: set[str], path: Path) -> None:
    """Raise ValueError if any barcodes are missing from the context."""
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(
            f"Missing benchmark metadata for {len(missing)} plate(s) in {path}: {preview}"
        )


# --- cpjump1_official_representation ---


def _classify_representation(context: OfficialSplitContext) -> str:
    """Classify a well into train/validate/test for official representation."""
    experiment_type = context.experiment_type.lower()
    if experiment_type in {"crispr", "orf"}:
        return "train"
    if experiment_type == "compound" and context.timepoint_code == "low":
        return "validate"
    if experiment_type == "compound" and context.timepoint_code == "high":
        return "test"
    raise ValueError(
        "Unsupported official representation split context: "
        f"{context.experiment_type} {context.timepoint_code}"
    )


def build_official_representation_subsets(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """CRISPR/ORF -> train, compound low -> validate, compound high -> test."""
    contexts = load_official_split_contexts()
    wells, missing = _iterate_official_wells(dataset, contexts)
    _raise_for_missing_wells(missing, resolve_official_split_metadata_path())

    subsets: dict[str, list[int]] = {"train": [], "validate": [], "test": []}
    for i, context, _info in wells:
        subsets[_classify_representation(context)].append(i)
    return subsets


def build_official_representation_groups(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """Group keys for the official representation strategy."""
    contexts = load_official_split_contexts()
    wells, missing = _iterate_official_wells(dataset, contexts)
    _raise_for_missing_wells(missing, resolve_official_split_metadata_path())

    grouped: dict[str, list[int]] = defaultdict(list)
    for i, context, info in wells:
        subset = _classify_representation(context)
        group_key = "::".join(
            [
                subset,
                context.cell_line,
                context.experiment_type,
                context.timepoint_code,
                info.broad_sample,
            ]
        )
        grouped[group_key].append(i)
    return dict(grouped)


# --- cpjump1_official_gene_compound ---


def build_official_gene_compound_subsets(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """60/20/20 split on unique 'across' targets."""
    contexts = load_official_split_contexts()
    wells, missing = _iterate_official_wells(dataset, contexts)
    _raise_for_missing_wells(missing, resolve_official_split_metadata_path())

    target_to_indices: dict[str, list[int]] = defaultdict(list)
    target_to_radix: dict[str, int | None] = {}

    for i, context, _info in wells:
        if not context.target_is_across or not context.target:
            continue
        target_to_indices[context.target].append(i)
        target_to_radix.setdefault(context.target, context.target_radix)

    ordered_targets = sorted(
        target_to_indices,
        key=lambda target: (
            target_to_radix[target] is None,
            target_to_radix[target] if target_to_radix[target] is not None else 10**9,
            target,
        ),
    )

    n_targets = len(ordered_targets)
    n_train = int(n_targets * 0.60)
    n_validate = (n_targets * 20 + 99) // 100 if n_targets else 0
    n_test = n_targets - n_train - n_validate
    if n_test < 0:
        n_test = 0
        n_validate = n_targets - n_train

    train_targets = set(ordered_targets[:n_train])
    validate_targets = set(ordered_targets[n_train : n_train + n_validate])
    test_targets = set(ordered_targets[n_train + n_validate :])

    subsets: dict[str, list[int]] = {"train": [], "validate": [], "test": []}
    for target, indices in target_to_indices.items():
        if target in train_targets:
            subsets["train"].extend(indices)
        elif target in validate_targets:
            subsets["validate"].extend(indices)
        elif target in test_targets:
            subsets["test"].extend(indices)
        else:  # pragma: no cover - defensive
            raise AssertionError(f"Target {target!r} was not assigned to a subset")

    return subsets


def build_official_gene_compound_groups(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """Group keys for the gene-compound strategy."""
    contexts = load_official_split_contexts()
    wells, missing = _iterate_official_wells(dataset, contexts)
    _raise_for_missing_wells(missing, resolve_official_split_metadata_path())

    grouped: dict[str, list[int]] = defaultdict(list)
    for i, context, _info in wells:
        if not context.target_is_across or not context.target:
            continue
        grouped[context.target].append(i)
    return dict(grouped)


# --- cellclip_cpjump_style ---


def build_cellclip_cpjump_subsets(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """Reproduce the CellCLIP CP-JUMP1 split at the well-bag level.

    75/25 train/test split within each (Cell_type, Time, Perturbation) slice.
    """
    plate_contexts = load_plate_contexts()
    wells, missing = _iterate_plate_wells(dataset, plate_contexts)
    _raise_for_missing_barcodes(missing, resolve_metadata_path())

    slice_groups: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, context, info in wells:
        slice_key = (context.cell_type, context.timepoint, context.perturbation)
        slice_groups[slice_key][info.broad_sample].append(i)

    subsets: dict[str, list[int]] = {"train": [], "validate": [], "test": []}
    for treatment_groups in slice_groups.values():
        treatment_keys = sorted(treatment_groups.keys())
        train_size = int(0.75 * len(treatment_keys))
        train_keys = set(treatment_keys[:train_size])
        test_keys = set(treatment_keys[train_size:])

        for key in train_keys:
            subsets["train"].extend(treatment_groups[key])
        for key in test_keys:
            subsets["test"].extend(treatment_groups[key])

    return subsets


def build_cellclip_cpjump_groups(
    dataset: MorphoCLIPDataset,
) -> dict[str, list[int]]:
    """Group keys for the CellCLIP CP-JUMP1 strategy."""
    plate_contexts = load_plate_contexts()
    wells, missing = _iterate_plate_wells(dataset, plate_contexts)
    _raise_for_missing_barcodes(missing, resolve_metadata_path())

    grouped: dict[str, list[int]] = defaultdict(list)
    for i, context, info in wells:
        group_key = "::".join(
            [
                context.cell_type,
                context.perturbation,
                str(context.timepoint),
                info.broad_sample,
            ]
        )
        grouped[group_key].append(i)
    return dict(grouped)


# --- Dispatch + manifest ---


def _resolve_benchmark_split_indices(
    dataset: MorphoCLIPDataset,
    *,
    strategy: str,
) -> tuple[list[int], list[int], list[int]]:
    """Resolve benchmark split indices by strategy name."""
    if strategy == "cpjump1_official_representation":
        subsets = build_official_representation_subsets(dataset)
    elif strategy == "cpjump1_official_gene_compound":
        subsets = build_official_gene_compound_subsets(dataset)
    elif strategy == "cellclip_cpjump_style":
        subsets = build_cellclip_cpjump_subsets(dataset)
    else:
        raise ValueError(f"Unknown benchmark split strategy: {strategy!r}")
    return subsets["train"], subsets["validate"], subsets["test"]


def build_split_manifest(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a split manifest keyed by ``Metadata_Plate`` + ``Metadata_Well``."""
    del val_fraction, test_fraction, seed

    train_idx, val_idx, test_idx = _resolve_benchmark_split_indices(
        dataset,
        strategy=strategy,
    )
    subset_by_idx = {idx: "train" for idx in train_idx}
    subset_by_idx.update({idx: "validate" for idx in val_idx})
    subset_by_idx.update({idx: "test" for idx in test_idx})

    official_contexts: dict = {}
    try:
        official_contexts = load_official_split_contexts()
    except Exception:
        pass

    records: list[dict[str, str | int]] = []
    for idx, (plate, well, _) in enumerate(dataset.index_entries):
        subset = subset_by_idx.get(idx)
        if subset is None:
            continue

        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)
        context = official_contexts.get((barcode, well))
        records.append(
            {
                "split_strategy": strategy,
                "subset": subset,
                "Metadata_Plate": barcode,
                "Metadata_Well": well,
                "Metadata_broad_sample": info.broad_sample,
                "Metadata_cell_line": getattr(context, "cell_line", ""),
                "Metadata_experiment_type": getattr(context, "experiment_type", ""),
                "Metadata_timepoint": getattr(context, "timepoint", ""),
                "Metadata_timepoint_code": getattr(context, "timepoint_code", ""),
                "Metadata_target": getattr(context, "target", "") or info.gene or info.target_list,
            }
        )

    return pd.DataFrame.from_records(records)


def build_split_groups(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
) -> dict[str, list[int]]:
    """Build grouping keys for benchmark split strategies."""
    if strategy == "cpjump1_official_representation":
        return build_official_representation_groups(dataset)
    if strategy == "cpjump1_official_gene_compound":
        return build_official_gene_compound_groups(dataset)
    if strategy == "cellclip_cpjump_style":
        return build_cellclip_cpjump_groups(dataset)
    raise ValueError(f"Unknown benchmark split strategy: {strategy!r}")


def create_splits(
    dataset: MorphoCLIPDataset,
    strategy: str = "cpjump1_official_representation",
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test subsets using benchmark strategies."""
    train_idx, val_idx, test_idx = _resolve_benchmark_split_indices(
        dataset,
        strategy=strategy,
    )
    logger.info(
        "Benchmark split: train=%d, val=%d, test=%d (strategy=%s)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        strategy,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
