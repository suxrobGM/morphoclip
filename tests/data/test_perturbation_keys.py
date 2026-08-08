"""Tests for canonical target-gene keys."""

from morphoclip.data.perturbation import (
    PerturbationInfo,
    PerturbationType,
    parse_target_gene_key,
    target_gene_key,
)


def _compound(target_list: str) -> PerturbationInfo:
    return PerturbationInfo(
        pert_type=PerturbationType.COMPOUND,
        broad_sample="BRD-X",
        target_list=target_list,
    )


def _crispr(gene: str) -> PerturbationInfo:
    return PerturbationInfo(
        pert_type=PerturbationType.CRISPR,
        broad_sample="BRDN-X",
        gene=gene,
    )


class TestTargetGeneKey:
    """Tests for :func:`target_gene_key` parsing and canonicalization."""

    def test_compound_uses_pipe_delimited_target_list(self) -> None:
        assert target_gene_key(_compound("CACNA1A|CACNB4")) == "CACNA1A|CACNB4"

    def test_crispr_uses_gene(self) -> None:
        assert target_gene_key(_crispr("TP53")) == "TP53"

    def test_orf_uses_gene(self) -> None:
        info = PerturbationInfo(pert_type=PerturbationType.ORF, gene="MYC")
        assert target_gene_key(info) == "MYC"

    def test_comma_delimiter_tolerated(self) -> None:
        assert target_gene_key(_compound("GENEB, GENEA")) == "GENEA|GENEB"

    def test_mixed_delimiters(self) -> None:
        assert target_gene_key(_compound("GENEC|GENEA,GENEB")) == "GENEA|GENEB|GENEC"

    def test_case_normalized(self) -> None:
        assert target_gene_key(_compound("tp53|Brca1")) == "BRCA1|TP53"

    def test_whitespace_stripped(self) -> None:
        assert target_gene_key(_compound("  TP53 |  BRCA1  ")) == "BRCA1|TP53"

    def test_duplicates_removed(self) -> None:
        assert target_gene_key(_compound("TP53|tp53| TP53 ")) == "TP53"

    def test_sorted_order_is_canonical(self) -> None:
        assert target_gene_key(_compound("B|A")) == target_gene_key(_compound("A|B"))

    def test_empty_target_list(self) -> None:
        assert target_gene_key(_compound("")) == ""

    def test_only_delimiters_yields_empty(self) -> None:
        assert target_gene_key(_compound("| , |")) == ""

    def test_negcon_yields_empty(self) -> None:
        info = PerturbationInfo(pert_type=PerturbationType.NEGCON, gene="TP53")
        assert target_gene_key(info) == ""

    def test_unknown_type_yields_empty(self) -> None:
        info = PerturbationInfo(pert_type=PerturbationType.UNKNOWN, gene="TP53")
        assert target_gene_key(info) == ""

    def test_compound_ignores_gene_field(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            gene="TP53",
            target_list="",
        )
        assert target_gene_key(info) == ""


class TestParseTargetGeneKey:
    """Tests for round-tripping a canonical key back to a gene set."""

    def test_roundtrip(self) -> None:
        key = target_gene_key(_compound("GENEB|GENEA"))
        assert parse_target_gene_key(key) == frozenset({"GENEA", "GENEB"})

    def test_empty_key_is_empty_set(self) -> None:
        assert parse_target_gene_key("") == frozenset()

    def test_empty_set_never_intersects(self) -> None:
        empty = parse_target_gene_key("")
        assert not (empty & parse_target_gene_key("TP53"))
        assert not (empty & empty)
