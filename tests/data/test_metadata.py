"""Tests for morphoclip.data.metadata and morphoclip.data.perturbation modules.

Uses real CPJUMP1 metadata files from data/metadata/ when available.
"""

from pathlib import Path

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

BATCH = "2020_11_04_CPJUMP1"


class TestPerturbationType:
    def test_values(self) -> None:
        assert PerturbationType.COMPOUND == "compound"
        assert PerturbationType.CRISPR == "crispr"
        assert PerturbationType.ORF == "orf"
        assert PerturbationType.NEGCON == "negcon"
        assert PerturbationType.POSCON == "poscon"
        assert PerturbationType.UNKNOWN == "unknown"

    def test_is_str(self) -> None:
        assert isinstance(PerturbationType.COMPOUND, str)


class TestPerturbationInfo:
    def test_defaults(self) -> None:
        info = PerturbationInfo()
        assert info.pert_type == PerturbationType.UNKNOWN
        assert info.broad_sample == ""
        assert info.smiles == ""

    def test_frozen(self) -> None:
        info = PerturbationInfo(pert_type=PerturbationType.COMPOUND)
        with pytest.raises(AttributeError):
            info.pert_type = PerturbationType.CRISPR  # type: ignore[misc]

    def test_custom_fields(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            pert_iname="Aloxistatin",
            target_list="CTSL",
            smiles="CC(CC)C=O",
        )
        assert info.pert_iname == "Aloxistatin"
        assert info.target_list == "CTSL"


class TestWellConversion:
    @pytest.mark.parametrize(
        "row,col,expected",
        [
            (1, 1, "A01"),
            (1, 24, "A24"),
            (2, 12, "B12"),
            (16, 24, "P24"),
            (8, 5, "H05"),
        ],
    )
    def test_well_from_row_col(self, row: int, col: int, expected: str) -> None:
        assert well_from_row_col(row, col) == expected

    @pytest.mark.parametrize(
        "well,expected_row,expected_col",
        [
            ("A01", 1, 1),
            ("A24", 1, 24),
            ("B12", 2, 12),
            ("P24", 16, 24),
            ("H05", 8, 5),
            ("a01", 1, 1),  # lowercase
        ],
    )
    def test_row_col_from_well(self, well: str, expected_row: int, expected_col: int) -> None:
        assert row_col_from_well(well) == (expected_row, expected_col)

    def test_roundtrip(self) -> None:
        for row in range(1, 17):
            for col in range(1, 25):
                well = well_from_row_col(row, col)
                assert row_col_from_well(well) == (row, col)

    def test_invalid_row(self) -> None:
        with pytest.raises(ValueError, match="Row must be 1-16"):
            well_from_row_col(0, 1)
        with pytest.raises(ValueError, match="Row must be 1-16"):
            well_from_row_col(17, 1)

    def test_invalid_col(self) -> None:
        with pytest.raises(ValueError, match="Column must be 1-24"):
            well_from_row_col(1, 0)
        with pytest.raises(ValueError, match="Column must be 1-24"):
            well_from_row_col(1, 25)


class TestExtractPlateBarcode:
    def test_standard_format(self) -> None:
        assert extract_plate_barcode("BR00116991__2020-11-05T19_51_35-Measurement1") == "BR00116991"

    def test_barcode_only(self) -> None:
        assert extract_plate_barcode("BR00116991") == "BR00116991"

    def test_double_underscore(self) -> None:
        assert extract_plate_barcode("PLATE__extra__info") == "PLATE"


class TestTextTemplates:
    def test_compound_full(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            pert_iname="Aloxistatin",
            target_list="CTSL",
            moa="Cysteine protease inhibitor",
            smiles="CC(CC)C=O",
        )
        text = generate_text(info, level="full")
        assert "Chemical perturbation: Aloxistatin." in text
        assert "Target: CTSL." in text
        assert "Function: Cysteine protease inhibitor." in text
        assert "SMILES: CC(CC)C=O." in text

    def test_compound_name_only(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            pert_iname="Dasatinib",
        )
        text = generate_text(info, level="name_only")
        assert text == "Chemical perturbation: Dasatinib."

    def test_compound_name_target(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            pert_iname="Dasatinib",
            target_list="ABL1|SRC",
        )
        text = generate_text(info, level="name_target")
        assert "Chemical perturbation: Dasatinib." in text
        assert "Target: ABL1|SRC." in text

    def test_crispr_full(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.CRISPR,
            gene="TP53",
            protein_name="Tumor protein p53",
            moa="Tumor suppressor",
            go_terms="apoptotic process (GO:0006915)",
        )
        text = generate_text(info, level="full")
        assert "CRISPR knockout of TP53." in text
        assert "Protein: Tumor protein p53." in text
        assert "Function: Tumor suppressor." in text
        assert "GO terms:" in text

    def test_crispr_name_only(self) -> None:
        info = PerturbationInfo(pert_type=PerturbationType.CRISPR, gene="TP53")
        text = generate_text(info, level="name_only")
        assert text == "CRISPR knockout of TP53."

    def test_orf_full(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.ORF,
            gene="BRCA1",
            protein_name="Breast cancer type 1",
        )
        text = generate_text(info, level="full")
        assert "ORF overexpression of BRCA1." in text
        assert "Protein: Breast cancer type 1." in text

    def test_negcon(self) -> None:
        info = PerturbationInfo(pert_type=PerturbationType.NEGCON)
        text = generate_text(info, level="full")
        assert "Negative control" in text

    def test_poscon(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.POSCON,
            control_type="poscon_diverse",
        )
        text = generate_text(info, level="full")
        assert "Positive control" in text
        assert "poscon_diverse" in text

    def test_unknown_type(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.UNKNOWN,
            broad_sample="MYSTERY-001",
        )
        text = generate_text(info, level="full")
        assert "Unknown perturbation" in text

    def test_compound_missing_annotations_falls_back(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            pert_iname="SomeCompound",
        )
        text = generate_text(info, level="full")
        assert text == "Chemical perturbation: SomeCompound."


class TestMetadataIndex:
    """Tests using real metadata files from data/metadata/."""

    def test_from_directory(self, metadata_dir: Path) -> None:
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        plates = index.plates()
        assert len(plates) >= 50
        assert "BR00116991" in plates

    def test_plate_types(self, metadata_dir: Path) -> None:
        """Verify plates map to all three platemap types."""
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        compound_info = index.lookup("BR00116991", "A01")
        assert compound_info.pert_type == PerturbationType.COMPOUND

        crispr_info = index.lookup("BR00117000", "A01")
        assert crispr_info.pert_type == PerturbationType.CRISPR

        orf_info = index.lookup("BR00117006", "A01")
        assert orf_info.pert_type == PerturbationType.ORF

    def test_lookup_real_compound(self, metadata_dir: Path) -> None:
        """gabapentin-enacarbil at BR00116991 A01."""
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        info = index.lookup("BR00116991", "A01")
        assert info.pert_type == PerturbationType.COMPOUND
        assert info.broad_sample == "BRD-A86665761-001-01-1"
        assert info.pert_iname == "gabapentin-enacarbil"
        assert "CACNB4" in info.target_list

    def test_lookup_real_crispr(self, metadata_dir: Path) -> None:
        """HIF1A at BR00117000 A01."""
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        info = index.lookup("BR00117000", "A01")
        assert info.pert_type == PerturbationType.CRISPR
        assert info.gene == "HIF1A"

    def test_lookup_real_orf(self, metadata_dir: Path) -> None:
        """KCNN1 at BR00117006 A01."""
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        info = index.lookup("BR00117006", "A01")
        assert info.pert_type == PerturbationType.ORF
        assert info.gene == "KCNN1"

    def test_dmso_control(self, metadata_dir: Path) -> None:
        """A02 on compound platemap has empty broad_sample = DMSO control."""
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        info = index.lookup("BR00116991", "A02")
        assert info.pert_type == PerturbationType.NEGCON

    def test_wells_for_plate(self, metadata_dir: Path) -> None:
        """Each plate should have 384 wells (16 rows x 24 cols)."""
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        wells = index.wells_for_plate("BR00116991")
        assert len(wells) == 384
        assert "A01" in wells
        assert "P24" in wells

    def test_lookup_unknown_well(self, metadata_dir: Path) -> None:
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        info = index.lookup("BR00116991", "Z99")
        assert info.pert_type == PerturbationType.UNKNOWN

    def test_lookup_unknown_plate(self, metadata_dir: Path) -> None:
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        info = index.lookup("NONEXISTENT", "A01")
        assert info.pert_type == PerturbationType.UNKNOWN

    def test_repr(self, metadata_dir: Path) -> None:
        index = MetadataIndex.from_directory(metadata_dir, batch=BATCH)
        r = repr(index)
        assert "MetadataIndex" in r
        assert "plates=" in r
