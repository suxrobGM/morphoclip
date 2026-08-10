"""Tests for the split strategies in morphoclip.splits.

Uses the committed CPJUMP1 metadata fixture plus synthesized .pt features, with
the two reference-metadata files the strategies read redirected at stand-ins.
"""

from collections.abc import Sequence
from pathlib import Path

import pytest

import morphoclip.splits.contexts as split_contexts
from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.metadata import MetadataIndex
from morphoclip.splits.api import build_split_groups, create_splits
from morphoclip.splits.manifest import build_split_manifest
from tests.support.constants import HIDDEN_DIM
from tests.support.features import BATCH, make_feature_root

OFFICIAL_CSV_ROWS = (
    "Metadata_Plate,Metadata_Well,Metadata_broad_sample,Metadata_target,"
    "Metadata_cell_line,Metadata_experiment_type,Metadata_timepoint,"
    "Metadata_timepoint_code,Metadata_target_is_across,Metadata_target_radix",
    "BR00117000,A04,BRDN0000259015,OPRL1,U2OS,CRISPR,144,high,TRUE,1",
    "BR00117000,A11,BRDN0000259016,OPRL1,U2OS,CRISPR,144,high,TRUE,1",
    "BR00116991,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,24,low,TRUE,2",
    "BR00117017,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,48,high,TRUE,2",
    "BR00117020,A01,ccsbBroad304_00900,KCNN1,A549,ORF,48,low,TRUE,3",
    "BR00117003,A01,BRDN0001480888,HIF1A,A549,CRISPR,144,high,TRUE,4",
)

REPRESENTATION_WELLS = (
    ("BR00117003", "A01"),
    ("BR00117020", "A01"),
    ("BR00116991", "A01"),
    ("BR00117017", "A01"),
)
GENE_COMPOUND_WELLS = (("BR00117000", "A04"), ("BR00117000", "A11"), *REPRESENTATION_WELLS)

# (plate, well, cell type). Wells A01 of 91/92/95 share one broad_sample, so the
# same treatment appears twice in the A549 slice and once in the U2OS slice.
CELLCLIP_PLATES = (
    ("BR00116991", "A01", "A549"),
    ("BR00116992", "A01", "A549"),
    ("BR00116993", "A03", "A549"),
    ("BR00116994", "A04", "A549"),
    ("BR00116995", "A01", "U2OS"),
)


def _dataset(
    tmp_path: Path, metadata: MetadataIndex, wells: Sequence[tuple[str, str]]
) -> MorphoCLIPDataset:
    plates: dict[str, list[str]] = {}
    for plate, well in wells:
        plates.setdefault(plate, []).append(well)
    make_feature_root(tmp_path, plates, dim=HIDDEN_DIM)
    return MorphoCLIPDataset(feature_dir=tmp_path, metadata=metadata, plates=list(plates))


def _index_map(dataset: MorphoCLIPDataset) -> dict[tuple[str, str], int]:
    return {(plate, well): i for i, (plate, well, _) in enumerate(dataset.index_entries)}


@pytest.fixture
def official_split_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the official CPJUMP1 split CSV at a six-well stand-in."""
    path = tmp_path / "cpjump1_metadata.csv"
    path.write_text("\n".join(OFFICIAL_CSV_ROWS) + "\n", encoding="utf-8")
    monkeypatch.setattr(split_contexts, "OFFICIAL_SPLIT_METADATA_PATH", path)


@pytest.fixture
def representation_dataset(
    tmp_path: Path, metadata_index: MetadataIndex, official_split_metadata: None
) -> MorphoCLIPDataset:
    return _dataset(tmp_path, metadata_index, REPRESENTATION_WELLS)


@pytest.fixture
def cellclip_experiment_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the benchmark experiment TSV at the plates the cellclip split slices on."""
    path = tmp_path / "experiment-metadata.tsv"
    header = "Batch\tPlate_Map_Name\tAssay_Plate_Barcode\tPerturbation\tCell_type\tTime"
    rows = [
        f"{BATCH}\tcompound\t{plate}\tcompound\t{cell_type}\t24"
        for plate, _well, cell_type in CELLCLIP_PLATES
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    monkeypatch.setattr(split_contexts, "METADATA_PATH", path)


def test_unknown_strategy_is_rejected(representation_dataset: MorphoCLIPDataset) -> None:
    with pytest.raises(ValueError, match="Unknown split strategy"):
        create_splits(representation_dataset, "invalid")


def test_official_representation_splits_by_modality_and_timepoint(
    representation_dataset: MorphoCLIPDataset, metadata_index: MetadataIndex
) -> None:
    index = _index_map(representation_dataset)
    train, validate, test = create_splits(representation_dataset, "cpjump1_official_representation")

    assert set(train.indices) == {index[("BR00117003", "A01")], index[("BR00117020", "A01")]}
    assert set(validate.indices) == {index[("BR00116991", "A01")]}
    assert set(test.indices) == {index[("BR00117017", "A01")]}

    broad_sample = metadata_index.lookup("BR00117017", "A01").broad_sample
    groups = build_split_groups(representation_dataset, "cpjump1_official_representation")
    assert groups[f"test::A549::Compound::high::{broad_sample}"] == [index[("BR00117017", "A01")]]


def test_official_gene_compound_keeps_each_target_in_one_subset(
    tmp_path: Path, metadata_index: MetadataIndex, official_split_metadata: None
) -> None:
    dataset = _dataset(tmp_path, metadata_index, GENE_COMPOUND_WELLS)
    index = _index_map(dataset)
    oprl1 = sorted([index[("BR00117000", "A04")], index[("BR00117000", "A11")]])
    cacnb4 = sorted([index[("BR00116991", "A01")], index[("BR00117017", "A01")]])

    train, validate, test = create_splits(dataset, "cpjump1_official_gene_compound")

    assert set(train.indices) == set(oprl1) | set(cacnb4)
    assert set(validate.indices) == {index[("BR00117020", "A01")]}
    assert set(test.indices) == {index[("BR00117003", "A01")]}

    groups = build_split_groups(dataset, "cpjump1_official_gene_compound")
    assert groups["OPRL1"] == oprl1
    assert groups["CACNB4"] == cacnb4


def test_cellclip_cpjump_style_slices_by_plate_context_and_keeps_replicates_together(
    tmp_path: Path, metadata_index: MetadataIndex, cellclip_experiment_metadata: None
) -> None:
    dataset = _dataset(tmp_path, metadata_index, [(p, w) for p, w, _ in CELLCLIP_PLATES])
    index = _index_map(dataset)
    replicates = sorted([index[("BR00116991", "A01")], index[("BR00116992", "A01")]])
    shared = metadata_index.lookup("BR00116991", "A01").broad_sample

    groups = build_split_groups(dataset, "cellclip_cpjump_style")
    assert groups[f"A549::compound::24::{shared}"] == replicates
    assert groups[f"U2OS::compound::24::{shared}"] == [index[("BR00116995", "A01")]]

    train, validate, test = create_splits(dataset, "cellclip_cpjump_style")
    assert list(validate.indices) == []
    assert len(train.indices) + len(test.indices) == len(dataset)
    # 3 A549 treatments, 75% of them train: the shared one sorts last, into test.
    assert set(replicates) <= set(test.indices)


def test_split_manifest_carries_one_row_per_well_with_its_context(
    representation_dataset: MorphoCLIPDataset,
) -> None:
    manifest = build_split_manifest(representation_dataset, "cpjump1_official_representation")

    assert manifest[["Metadata_Plate", "Metadata_Well"]].duplicated().sum() == 0
    assert set(manifest["subset"]) == {"train", "validate", "test"}
    row = manifest.query("Metadata_Plate=='BR00117017' and Metadata_Well=='A01'").iloc[0]
    assert row["subset"] == "test"
    assert row["Metadata_timepoint_code"] == "high"
