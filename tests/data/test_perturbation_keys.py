"""Tests for canonical target-gene keys."""

import pytest

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


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (_compound("CACNA1A|CACNB4"), "CACNA1A|CACNB4"),
        (_compound("GENEB, GENEA"), "GENEA|GENEB"),
        (_compound("GENEC|GENEA,GENEB"), "GENEA|GENEB|GENEC"),
        (_compound("tp53|Brca1"), "BRCA1|TP53"),
        (_compound("  TP53 |  BRCA1  "), "BRCA1|TP53"),
        (_compound("TP53|tp53| TP53 "), "TP53"),
        (_compound(""), ""),
        (_compound("| , |"), ""),
        (PerturbationInfo(pert_type=PerturbationType.CRISPR, gene="TP53"), "TP53"),
        (PerturbationInfo(pert_type=PerturbationType.ORF, gene="MYC"), "MYC"),
        (PerturbationInfo(pert_type=PerturbationType.NEGCON, gene="TP53"), ""),
        (PerturbationInfo(pert_type=PerturbationType.UNKNOWN, gene="TP53"), ""),
        (PerturbationInfo(pert_type=PerturbationType.COMPOUND, gene="TP53"), ""),
    ],
    ids=[
        "pipes",
        "commas",
        "mixed-delimiters",
        "case",
        "whitespace",
        "duplicates",
        "empty",
        "delimiters-only",
        "crispr-gene",
        "orf-gene",
        "negcon",
        "unknown-type",
        "compound-ignores-gene-field",
    ],
)
def test_a_gene_set_canonicalizes_to_one_sorted_upper_case_key(
    info: PerturbationInfo, expected: str
) -> None:
    assert target_gene_key(info) == expected


def test_a_canonical_key_parses_back_to_its_gene_set() -> None:
    assert parse_target_gene_key(target_gene_key(_compound("GENEB|GENEA"))) == frozenset(
        {"GENEA", "GENEB"}
    )
    assert parse_target_gene_key("") == frozenset()
