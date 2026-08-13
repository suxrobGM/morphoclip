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
from morphoclip.splits.contexts import load_plate_conditions
from morphoclip.splits.manifest import build_split_manifest
from tests.support.constants import HIDDEN_DIM
from tests.support.contexts import write_official_split_csv
from tests.support.features import BATCH, make_feature_root

OFFICIAL_CSV_ROWS = (
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
    write_official_split_csv(monkeypatch, tmp_path, OFFICIAL_CSV_ROWS)


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


def test_plate_conditions_group_replicate_plates_under_one_key(
    official_split_metadata: None,
) -> None:
    """CWA offsets are relative to this key, so replicate plates must share one."""
    conditions = load_plate_conditions()

    assert conditions["BR00116991"] == "A549|Compound|24"
    assert conditions["BR00117017"] == "A549|Compound|48"
    assert conditions["BR00117000"] == "U2OS|CRISPR|144"


def test_plate_conditions_fall_back_to_the_experiment_tsv_on_every_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, official_split_metadata: None
) -> None:
    """Plates the official CSV lacks get a key that separates density, antibiotics,
    cell line, and time delay; plates in both sources keep the official key."""
    header = (
        "Batch\tAssay_Plate_Barcode\tPerturbation\tCell_type\tTime"
        "\tDensity\tAntibiotics\tCell_line\tTime_delay"
    )
    rows = (
        header,
        "b\tBR00116991\tcompound\tA549\t24\t100\tabsent\tParental\tDay0",
        "b\tBR00117008\tcompound\tA549\t48\t80\tabsent\tParental\tDay0",
        "b\tBR00117009\tcompound\tA549\t48\t80\tabsent\tParental\tDay0",
        "b\tBR00117050\tcompound\tA549\t48\t100\tabsent\tCas9\tDay0",
    )
    tsv = tmp_path / "experiment-metadata.tsv"
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(split_contexts, "METADATA_PATH", tsv)

    conditions = load_plate_conditions()

    assert conditions["BR00116991"] == "A549|Compound|24"
    assert conditions["BR00117008"] == conditions["BR00117009"]
    assert conditions["BR00117008"] != conditions["BR00117050"]


def test_plate_conditions_reject_an_experiment_tsv_missing_a_condition_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, official_split_metadata: None
) -> None:
    """Without the check, a renamed column zeroes the offset of every fallback plate."""
    rows = (
        "Batch\tAssay_Plate_Barcode\tPerturbation\tCell_type\tTime\tDensity\tAntibiotics",
        "b\tBR00117008\tcompound\tA549\t48\t80\tabsent",
    )
    tsv = tmp_path / "experiment-metadata.tsv"
    tsv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(split_contexts, "METADATA_PATH", tsv)

    with pytest.raises(ValueError, match="Cell_line, Time_delay"):
        load_plate_conditions()


def test_plate_conditions_reject_a_plate_whose_wells_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_official_split_csv(
        monkeypatch,
        tmp_path,
        (
            "BR00116991,A01,BRD-A86665761-001-01-1,CACNB4,A549,Compound,24,low,TRUE,2",
            "BR00116991,A02,BRD-A86665761-001-01-1,CACNB4,U2OS,Compound,24,low,TRUE,2",
        ),
    )

    with pytest.raises(ValueError, match="conflicting conditions"):
        load_plate_conditions()


def test_split_manifest_carries_one_row_per_well_with_its_context(
    representation_dataset: MorphoCLIPDataset,
) -> None:
    manifest = build_split_manifest(representation_dataset, "cpjump1_official_representation")

    assert manifest[["Metadata_Plate", "Metadata_Well"]].duplicated().sum() == 0
    assert set(manifest["subset"]) == {"train", "validate", "test"}
    row = manifest.query("Metadata_Plate=='BR00117017' and Metadata_Well=='A01'").iloc[0]
    assert row["subset"] == "test"
    assert row["Metadata_timepoint_code"] == "high"
