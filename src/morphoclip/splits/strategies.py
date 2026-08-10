"""Split-strategy builders: train/validate/test subsets and grouping keys.

Each strategy annotates dataset wells and produces index subsets, optionally
alongside grouping keys. :data:`STRATEGIES` maps a name to its builders and
drives dispatch in :mod:`morphoclip.splits.api`.

``pert_type`` needs only local metadata. The rest read the CPJUMP1 reference
metadata under ``data/reference/cpjump1/`` or the benchmark experiment TSV, and
their partition is fixed by that metadata, so they ignore :class:`SplitParams`.
"""

import hashlib
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.perturbation import (
    PerturbationInfo,
    extract_plate_barcode,
    is_control_or_empty,
)
from morphoclip.splits.contexts import (
    OfficialSplitContext,
    load_official_split_contexts,
    load_plate_contexts,
    resolve_metadata_path,
    resolve_official_split_metadata_path,
)


@dataclass(frozen=True)
class SplitParams:
    """Knobs for strategies that partition at random rather than by metadata."""

    val_fraction: float = 0.1
    seed: int = 42


def build_pert_type_subsets(
    dataset: MorphoCLIPDataset, params: SplitParams
) -> dict[str, list[int]]:
    """Split wells by hashing ``broad_sample``, stratified across modalities.

    Every perturbation type reaches every subset, and wells sharing a
    ``broad_sample`` always land together. ``test`` gets the same fraction as
    ``validate``.

    The md5 bucketing is a deliberately stable hash: changing it repartitions
    every split that already exists on disk.
    """
    sample_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (plate, well, _) in enumerate(dataset.index_entries):
        barcode = extract_plate_barcode(plate)
        info = dataset.metadata.lookup(barcode, well)
        if is_control_or_empty(info):
            continue
        sample_to_indices[info.broad_sample].append(i)

    subsets: dict[str, list[int]] = {"train": [], "validate": [], "test": []}
    for sample in sorted(sample_to_indices):
        digest = hashlib.md5(f"{params.seed}:{sample}".encode()).hexdigest()
        fraction = int(digest[:8], 16) / 0xFFFFFFFF
        if fraction < params.val_fraction:
            subset = "validate"
        elif fraction < 2 * params.val_fraction:
            subset = "test"
        else:
            subset = "train"
        subsets[subset].extend(sample_to_indices[sample])
    return subsets


def _iterate_wells[K, C](
    dataset: MorphoCLIPDataset,
    contexts: dict[K, C],
    *,
    key_fn: Callable[[str, str], K],
) -> tuple[list[tuple[int, C, PerturbationInfo]], set[K]]:
    """Annotate dataset wells with their context, filtering controls.

    Returns ``(annotated_wells, missing_keys)`` where each annotated entry is
    ``(index, context, pert_info)``.
    """
    annotated: list[tuple[int, C, PerturbationInfo]] = []
    missing: set[K] = set()

    for i, (plate, well, _) in enumerate(dataset.index_entries):
        barcode = extract_plate_barcode(plate)
        key = key_fn(barcode, well)
        context = contexts.get(key)
        if context is None:
            missing.add(key)
            continue
        info = dataset.metadata.lookup(barcode, well)
        if is_control_or_empty(info):
            continue
        annotated.append((i, context, info))

    return annotated, missing


def _raise_for_missing(
    missing: set[Any],
    path: Path,
    *,
    noun: str,
    metadata_label: str,
    formatter: Callable[[Any], str],
) -> None:
    """Raise ``ValueError`` if any context keys are missing."""
    if missing:
        preview = ", ".join(formatter(item) for item in sorted(missing)[:5])
        raise ValueError(f"Missing {metadata_label} for {len(missing)} {noun} in {path}: {preview}")


def _annotated_official_wells(
    dataset: MorphoCLIPDataset,
) -> list[tuple[int, OfficialSplitContext, PerturbationInfo]]:
    """Load official contexts and annotate dataset wells (raises on missing)."""
    contexts = load_official_split_contexts()
    wells, missing = _iterate_wells(dataset, contexts, key_fn=lambda barcode, well: (barcode, well))
    _raise_for_missing(
        missing,
        resolve_official_split_metadata_path(),
        noun="well(s)",
        metadata_label="official split metadata",
        formatter=lambda item: f"{item[0]}/{item[1]}",
    )
    return wells


def _annotated_plate_wells(dataset: MorphoCLIPDataset):
    """Load plate contexts and annotate dataset wells (raises on missing)."""
    plate_contexts = load_plate_contexts()
    wells, missing = _iterate_wells(dataset, plate_contexts, key_fn=lambda barcode, _well: barcode)
    _raise_for_missing(
        missing,
        resolve_metadata_path(),
        noun="plate(s)",
        metadata_label="benchmark metadata",
        formatter=str,
    )
    return wells


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
    dataset: MorphoCLIPDataset, _params: SplitParams
) -> dict[str, list[int]]:
    """CRISPR/ORF -> train, compound low -> validate, compound high -> test."""
    subsets: dict[str, list[int]] = {"train": [], "validate": [], "test": []}
    for i, context, _info in _annotated_official_wells(dataset):
        subsets[_classify_representation(context)].append(i)
    return subsets


def build_official_representation_groups(dataset: MorphoCLIPDataset) -> dict[str, list[int]]:
    """Group keys for the official representation strategy."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, context, info in _annotated_official_wells(dataset):
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


def build_official_gene_compound_subsets(
    dataset: MorphoCLIPDataset, _params: SplitParams
) -> dict[str, list[int]]:
    """60/20/20 split on unique 'across' targets."""
    target_to_indices: dict[str, list[int]] = defaultdict(list)
    target_to_radix: dict[str, int | None] = {}

    for i, context, _info in _annotated_official_wells(dataset):
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


def build_official_gene_compound_groups(dataset: MorphoCLIPDataset) -> dict[str, list[int]]:
    """Group keys for the gene-compound strategy."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, context, _info in _annotated_official_wells(dataset):
        if not context.target_is_across or not context.target:
            continue
        grouped[context.target].append(i)
    return dict(grouped)


def build_cellclip_cpjump_subsets(
    dataset: MorphoCLIPDataset, _params: SplitParams
) -> dict[str, list[int]]:
    """Reproduce the CellCLIP CP-JUMP1 split at the well-bag level.

    75/25 train/test split within each (Cell_type, Time, Perturbation) slice.
    """
    slice_groups: dict[tuple[str, int, str], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for i, context, info in _annotated_plate_wells(dataset):
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


def build_cellclip_cpjump_groups(dataset: MorphoCLIPDataset) -> dict[str, list[int]]:
    """Group keys for the CellCLIP CP-JUMP1 strategy."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, context, info in _annotated_plate_wells(dataset):
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


@dataclass(frozen=True)
class Strategy:
    """How one strategy partitions a dataset, and how it groups wells.

    ``groups`` is None for strategies with no grouping notion; asking for their
    groups is an error rather than an empty dict.
    """

    subsets: Callable[[MorphoCLIPDataset, SplitParams], dict[str, list[int]]]
    groups: Callable[[MorphoCLIPDataset], dict[str, list[int]]] | None = None


STRATEGIES: dict[str, Strategy] = {
    "pert_type": Strategy(build_pert_type_subsets),
    "cpjump1_official_representation": Strategy(
        build_official_representation_subsets,
        build_official_representation_groups,
    ),
    "cpjump1_official_gene_compound": Strategy(
        build_official_gene_compound_subsets,
        build_official_gene_compound_groups,
    ),
    "cellclip_cpjump_style": Strategy(
        build_cellclip_cpjump_subsets,
        build_cellclip_cpjump_groups,
    ),
}
