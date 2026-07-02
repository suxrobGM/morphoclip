"""Tests for the text encoder model components."""

import torch

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType
from morphoclip.models.projection_head import ProjectionHead
from morphoclip.models.prompts import (
    TEMPLATES,
    build_prompt,
    build_prompt_from_info,
    build_prompts,
    build_prompts_from_info,
    extract_template_fields,
)


class TestTemplates:
    def test_default_templates_exist(self) -> None:
        assert "{compound_name}" in TEMPLATES["compound"]
        assert "{gene_symbol}" in TEMPLATES["crispr"]
        assert "{gene_symbol}" in TEMPLATES["orf"]
        assert "{cell_line}" in TEMPLATES["negcon"]

    def test_extract_fields(self) -> None:
        fields = extract_template_fields("Hello {name}, your {item} is ready.")
        assert fields == ["name", "item"]


class TestBuildPrompt:
    def test_compound(self) -> None:
        prompt = build_prompt(
            {
                "modality": "compound",
                "compound_name": "PFI-1",
                "smiles": "CC(=O)Nc1ccc(F)cc1",
                "target_gene": "BRD4",
                "gene_function": "Bromodomain protein",
                "cell_line": "U2OS",
            }
        )
        assert "PFI-1" in prompt
        assert "BRD4" in prompt
        assert "U2OS" in prompt
        assert "SMILES" in prompt

    def test_crispr(self) -> None:
        prompt = build_prompt(
            {
                "modality": "crispr",
                "gene_symbol": "TP53",
                "gene_description": "Tumor protein p53",
                "gene_function": "Tumor suppressor",
                "cell_line": "A549",
            }
        )
        assert "TP53" in prompt
        assert "CRISPR" in prompt

    def test_orf(self) -> None:
        prompt = build_prompt(
            {
                "modality": "orf",
                "gene_symbol": "MYC",
                "gene_description": "MYC proto-oncogene",
                "gene_function": "Transcription factor",
                "cell_line": "U2OS",
            }
        )
        assert "MYC" in prompt
        assert "overexpressing" in prompt

    def test_negcon(self) -> None:
        prompt = build_prompt({"modality": "negcon", "cell_line": "U2OS"})
        assert "DMSO" in prompt
        assert "U2OS" in prompt

    def test_missing_fields_default_to_unknown(self) -> None:
        prompt = build_prompt({"modality": "compound"})
        assert "unknown" in prompt

    def test_from_info(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.COMPOUND,
            broad_sample="BRD-K12345",
            pert_iname="PFI-1",
            smiles="CC(=O)Nc1ccc(F)cc1",
            target_list="BRD4",
            moa="Bromodomain inhibitor",
            cell_line="U2OS",
        )
        prompt = build_prompt_from_info(info)
        assert "PFI-1" in prompt
        assert "BRD4" in prompt

    def test_from_info_crispr(self) -> None:
        info = PerturbationInfo(
            pert_type=PerturbationType.CRISPR,
            gene="TP53",
            protein_name="Tumor protein p53",
        )
        prompt = build_prompt_from_info(info)
        assert "TP53" in prompt
        assert "CRISPR" in prompt

    def test_batch(self) -> None:
        prompts = build_prompts(
            [
                {"modality": "compound", "compound_name": "A"},
                {"modality": "crispr", "gene_symbol": "B"},
            ]
        )
        assert len(prompts) == 2

    def test_batch_from_info(self) -> None:
        infos = [
            PerturbationInfo(pert_type=PerturbationType.COMPOUND, pert_iname="X"),
            PerturbationInfo(pert_type=PerturbationType.NEGCON),
        ]
        prompts = build_prompts_from_info(infos)
        assert len(prompts) == 2


class TestProjectionHead:
    def test_output_shape(self) -> None:
        head = ProjectionHead(input_dim=768, hidden_dim=256, output_dim=512)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 512)

    def test_l2_normalized(self) -> None:
        head = ProjectionHead(input_dim=768, output_dim=512)
        x = torch.randn(3, 768)
        out = head(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_different_dims(self) -> None:
        head = ProjectionHead(input_dim=1024, hidden_dim=128, output_dim=256)
        x = torch.randn(2, 1024)
        out = head(x)
        assert out.shape == (2, 256)
