"""Prompt templates and builder for BioClinical ModernBERT input.

Prompts are deliberately verbose: more description gives BERT more to embed.
The concise templates in ``morphoclip.data.perturbation`` are a separate
system, used only for human-readable dataset labels.
"""

import re

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType

TEMPLATES: dict[str, str] = {
    "compound": (
        "Cell Painting morphological profile of {cell_line} cells "
        "treated with the compound {compound_name}. "
        "Chemical structure (SMILES): {smiles}. "
        "Known target gene: {target_gene}. "
        "Target protein function: {gene_function}. "
        "Perturbation modality: chemical compound."
    ),
    "crispr": (
        "Cell Painting morphological profile of {cell_line} cells "
        "with CRISPR-Cas9 knockout of gene {gene_symbol}. "
        "Gene description: {gene_description}. "
        "Gene function: {gene_function}. "
        "Perturbation modality: CRISPR knockout."
    ),
    "orf": (
        "Cell Painting morphological profile of {cell_line} cells "
        "overexpressing gene {gene_symbol} via open reading frame construct. "
        "Gene description: {gene_description}. "
        "Gene function: {gene_function}. "
        "Perturbation modality: ORF overexpression."
    ),
    "negcon": (
        "Cell Painting morphological profile of {cell_line} cells "
        "treated with DMSO vehicle control. "
        "No active perturbation applied."
    ),
}

MODALITY_ALIASES: dict[str, str] = {
    "compound": "compound",
    "crispr": "crispr",
    "orf": "orf",
    "negcon": "negcon",
    "dmso": "negcon",
    "control": "negcon",
}

_FIELD_PATTERN = re.compile(r"\{(\w+)\}")

_PERT_TYPE_TO_MODALITY: dict[PerturbationType, str] = {
    PerturbationType.COMPOUND: "compound",
    PerturbationType.CRISPR: "crispr",
    PerturbationType.ORF: "orf",
    PerturbationType.NEGCON: "negcon",
    PerturbationType.POSCON: "negcon",
    PerturbationType.UNKNOWN: "compound",
}


def extract_template_fields(template: str) -> list[str]:
    """Extract {field_name} placeholders from a template string."""
    return _FIELD_PATTERN.findall(template)


def build_prompt(metadata: dict, templates: dict[str, str] | None = None) -> str:
    """Build a single prompt from a metadata dict."""
    tpl = templates or TEMPLATES
    modality = metadata.get("modality", "compound").lower().strip()
    template = tpl[MODALITY_ALIASES.get(modality, "compound")]

    prompt = template
    for key in _FIELD_PATTERN.findall(template):
        value = (metadata.get(key) or "").strip()
        prompt = prompt.replace("{" + key + "}", value or "unknown")

    return " ".join(prompt.split())


def build_prompts(metadata_list: list[dict], templates: dict[str, str] | None = None) -> list[str]:
    """Build prompts for a batch of metadata dicts."""
    return [build_prompt(m, templates) for m in metadata_list]


def prompt_metadata_from_info(info: PerturbationInfo) -> dict[str, str]:
    """The metadata dict :func:`build_prompt` reads, filled from a ``PerturbationInfo``.

    Also carries ``broad_sample`` so the text cache can key the entry by
    perturbation id.
    """
    return {
        "modality": _PERT_TYPE_TO_MODALITY[info.pert_type],
        "broad_sample": info.broad_sample,
        "compound_name": info.pert_iname,
        "smiles": info.smiles,
        "target_gene": info.target_list,
        "gene_function": info.moa,
        "gene_symbol": info.gene,
        "gene_description": info.protein_name,
        "cell_line": info.cell_line,
    }


def build_prompt_from_info(info: PerturbationInfo, templates: dict[str, str] | None = None) -> str:
    """Build a prompt from a ``PerturbationInfo`` dataclass."""
    return build_prompt(prompt_metadata_from_info(info), templates)
