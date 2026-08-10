"""Tests for the prompt builders and the text projection head."""

import pytest
import torch

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType
from morphoclip.models.projection_head import ProjectionHead
from morphoclip.models.prompts import (
    build_prompt,
    build_prompt_from_info,
    extract_template_fields,
)


def test_extract_template_fields_returns_placeholders_in_order() -> None:
    assert extract_template_fields("Hello {name}, your {item} is ready.") == ["name", "item"]


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {
                "modality": "compound",
                "compound_name": "PFI-1",
                "smiles": "CC(=O)Nc1ccc(F)cc1",
                "target_gene": "BRD4",
                "gene_function": "Bromodomain protein",
                "cell_line": "U2OS",
            },
            ("PFI-1", "BRD4", "U2OS", "SMILES"),
        ),
        (
            {
                "modality": "crispr",
                "gene_symbol": "TP53",
                "gene_description": "Tumor protein p53",
                "cell_line": "A549",
            },
            ("TP53", "CRISPR", "A549"),
        ),
        (
            {
                "modality": "orf",
                "gene_symbol": "MYC",
                "gene_description": "MYC proto-oncogene",
                "cell_line": "U2OS",
            },
            ("MYC", "overexpressing", "U2OS"),
        ),
        ({"modality": "negcon", "cell_line": "U2OS"}, ("DMSO", "U2OS")),
        ({"modality": "compound"}, ("unknown",)),
    ],
)
def test_each_modality_renders_its_own_template(metadata: dict, expected: tuple[str, ...]) -> None:
    prompt = build_prompt(metadata)
    for fragment in expected:
        assert fragment in prompt


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (
            PerturbationInfo(
                pert_type=PerturbationType.COMPOUND,
                broad_sample="BRD-K12345",
                pert_iname="PFI-1",
                smiles="CC(=O)Nc1ccc(F)cc1",
                target_list="BRD4",
                moa="Bromodomain inhibitor",
                cell_line="U2OS",
            ),
            ("PFI-1", "BRD4", "U2OS"),
        ),
        (
            PerturbationInfo(
                pert_type=PerturbationType.CRISPR, gene="TP53", protein_name="Tumor protein p53"
            ),
            ("TP53", "CRISPR"),
        ),
    ],
)
def test_perturbation_info_fields_reach_the_template(
    info: PerturbationInfo, expected: tuple[str, ...]
) -> None:
    prompt = build_prompt_from_info(info)
    for fragment in expected:
        assert fragment in prompt


def test_projection_head_returns_unit_norm_rows_of_the_output_width() -> None:
    out = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=512)(torch.randn(4, 768))
    assert out.shape == (4, 512)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
