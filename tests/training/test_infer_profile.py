"""Unit tests for morphoclip.cli.infer._profile_row."""

from morphoclip.cli.infer import _profile_row
from morphoclip.data.perturbation import PerturbationInfo, PerturbationType


class TestProfileRow:
    def test_compound_row_has_expected_keys_and_values(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            broad_sample="BRD-K12345",
            target_list="EGFR",
            gene="",
            control_type="",
        )

        row = _profile_row("BR00116991", "A01", info)

        assert set(row.keys()) == {
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_broad_sample",
            "Metadata_pert_type",
            "Metadata_target",
            "Metadata_gene",
            "Metadata_control_type",
        }
        assert row["Metadata_Plate"] == "BR00116991"
        assert row["Metadata_Well"] == "A01"
        assert row["Metadata_broad_sample"] == "BRD-K12345"
        assert row["Metadata_pert_type"] == "COMPOUND"
        assert row["Metadata_target"] == "EGFR"
        assert row["Metadata_gene"] == ""
        assert row["Metadata_control_type"] == ""

    def test_crispr_row_uses_gene_field(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.CRISPR,
            broad_sample="crispr-EGFR",
            gene="EGFR",
        )

        row = _profile_row("BR00116991", "B02", info)

        assert row["Metadata_gene"] == "EGFR"
        assert row["Metadata_target"] == ""
        assert row["Metadata_pert_type"] == "CRISPR"

    def test_negcon_row_reports_control_type(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.NEGCON,
            broad_sample="",
            control_type="DMSO",
        )

        row = _profile_row("BR00116991", "P24", info)

        assert row["Metadata_control_type"] == "DMSO"
        assert row["Metadata_pert_type"] == "NEGCON"
        assert row["Metadata_broad_sample"] == ""

    def test_no_attribute_error_for_default_fields(self) -> None:
        # PerturbationInfo has `target_list`/`gene`, not `target`/`gene_symbol`.
        info = PerturbationInfo()

        row = _profile_row("PLATE1", "A01", info)

        assert row["Metadata_target"] == ""
        assert row["Metadata_gene"] == ""
        assert row["Metadata_control_type"] == ""
