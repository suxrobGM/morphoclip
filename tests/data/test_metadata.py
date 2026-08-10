"""Tests for morphoclip.data.metadata and morphoclip.data.perturbation.

The lookup cases pin values read out of the committed CPJUMP1 platemap fixture,
so a change to the platemap parser shows up as a wrong perturbation, not just a
missing key.
"""

import pytest

from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import (
    PerturbationInfo,
    PerturbationType,
    extract_plate_barcode,
    generate_text,
    row_col_from_well,
    well_from_row_col,
)

COMPOUND = PerturbationInfo(
    pert_type=PerturbationType.COMPOUND,
    pert_iname="Aloxistatin",
    target_list="CTSL",
    moa="Cysteine protease inhibitor",
    smiles="CC(CC)C=O",
)
CRISPR = PerturbationInfo(
    pert_type=PerturbationType.CRISPR,
    gene="TP53",
    protein_name="Tumor protein p53",
    moa="Tumor suppressor",
    go_terms="apoptotic process (GO:0006915)",
)


@pytest.mark.parametrize(
    ("row", "col", "expected"),
    [(1, 1, "A01"), (1, 24, "A24"), (8, 5, "H05"), (16, 24, "P24")],
)
def test_a_row_and_column_render_as_a_384_well_label(row: int, col: int, expected: str) -> None:
    assert well_from_row_col(row, col) == expected


def test_every_well_on_the_plate_round_trips_back_to_its_row_and_column() -> None:
    for row in range(1, 17):
        for col in range(1, 25):
            assert row_col_from_well(well_from_row_col(row, col)) == (row, col)
    assert row_col_from_well("a01") == (1, 1)


@pytest.mark.parametrize(
    ("row", "col", "match"),
    [(0, 1, "Row must be 1-16"), (17, 1, "Row must be 1-16"), (1, 0, "Column"), (1, 25, "Column")],
)
def test_a_position_off_the_plate_is_rejected(row: int, col: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        well_from_row_col(row, col)


@pytest.mark.parametrize(
    ("plate_dir_name", "expected"),
    [
        ("BR00116991__2020-11-05T19_51_35-Measurement1", "BR00116991"),
        ("BR00116991", "BR00116991"),
        ("PLATE__extra__info", "PLATE"),
    ],
)
def test_a_plate_directory_name_yields_its_barcode(plate_dir_name: str, expected: str) -> None:
    assert extract_plate_barcode(plate_dir_name) == expected


@pytest.mark.parametrize(
    ("info", "level", "expected"),
    [
        (
            COMPOUND,
            "full",
            "Chemical perturbation: Aloxistatin. Target: CTSL. "
            "Function: Cysteine protease inhibitor. SMILES: CC(CC)C=O.",
        ),
        (COMPOUND, "name_target", "Chemical perturbation: Aloxistatin. Target: CTSL."),
        (COMPOUND, "name_only", "Chemical perturbation: Aloxistatin."),
        (
            CRISPR,
            "full",
            "CRISPR knockout of TP53. Protein: Tumor protein p53. Function: Tumor suppressor. "
            "GO terms: apoptotic process (GO:0006915).",
        ),
        (
            PerturbationInfo(
                pert_type=PerturbationType.ORF, gene="BRCA1", protein_name="Breast cancer type 1"
            ),
            "full",
            "ORF overexpression of BRCA1. Protein: Breast cancer type 1.",
        ),
        (
            PerturbationInfo(pert_type=PerturbationType.COMPOUND, pert_iname="SomeCompound"),
            "full",
            "Chemical perturbation: SomeCompound.",
        ),
        (PerturbationInfo(pert_type=PerturbationType.NEGCON), "full", "Negative control (DMSO)."),
        (
            PerturbationInfo(pert_type=PerturbationType.POSCON, control_type="poscon_diverse"),
            "name_only",
            "Positive control (poscon_diverse).",
        ),
        (
            PerturbationInfo(pert_type=PerturbationType.UNKNOWN, broad_sample="MYSTERY-001"),
            "full",
            "Unknown perturbation: MYSTERY-001.",
        ),
    ],
)
def test_generate_text_renders_the_fields_its_level_allows(
    info: PerturbationInfo, level: str, expected: str
) -> None:
    assert generate_text(info, level=level) == expected


@pytest.mark.parametrize(
    ("plate", "well", "expected"),
    [
        (
            "BR00116991",
            "A01",
            {
                "pert_type": PerturbationType.COMPOUND,
                "broad_sample": "BRD-A86665761-001-01-1",
                "pert_iname": "gabapentin-enacarbil",
                "target_list": "CACNB4",
            },
        ),
        ("BR00117000", "A01", {"pert_type": PerturbationType.CRISPR, "gene": "HIF1A"}),
        ("BR00117006", "A01", {"pert_type": PerturbationType.ORF, "gene": "KCNN1"}),
        ("BR00116991", "A02", {"pert_type": PerturbationType.NEGCON, "broad_sample": ""}),
        ("BR00116991", "Z99", {"pert_type": PerturbationType.UNKNOWN}),
        ("NONEXISTENT", "A01", {"pert_type": PerturbationType.UNKNOWN}),
    ],
)
def test_lookup_resolves_a_well_to_its_annotated_perturbation(
    metadata_index: MetadataIndex, plate: str, well: str, expected: dict[str, object]
) -> None:
    info = metadata_index.lookup(plate, well)
    assert {field: getattr(info, field) for field in expected} == expected


def test_the_index_covers_every_plate_and_every_well_on_a_plate(
    metadata_index: MetadataIndex,
) -> None:
    assert len(metadata_index.plates()) == 51
    wells = metadata_index.wells_for_plate("BR00116991")
    assert len(wells) == 384
    assert {"A01", "P24"} <= set(wells)
