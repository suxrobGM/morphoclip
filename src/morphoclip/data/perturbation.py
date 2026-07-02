"""Perturbation data types, text templates, and well position helpers."""

from dataclasses import dataclass
from enum import StrEnum


class PerturbationType(StrEnum):
    """Perturbation modality in CPJUMP1."""

    COMPOUND = "compound"
    CRISPR = "crispr"
    ORF = "orf"
    NEGCON = "negcon"
    POSCON = "poscon"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class PerturbationInfo:
    """Perturbation metadata for a single well.

    All string fields default to empty when annotation is unavailable.
    """

    pert_type: PerturbationType = PerturbationType.UNKNOWN
    broad_sample: str = ""
    pert_iname: str = ""
    target_list: str = ""
    gene: str = ""
    smiles: str = ""
    pubchem_cid: str = ""
    moa: str = ""
    protein_name: str = ""
    go_terms: str = ""
    cell_line: str = ""
    control_type: str = ""
    target_sequence: str = ""
    negcon_control_type: str = ""


# Prefix and extra-fields table per perturbation type.
# Each entry: (prefix_template, list of (field_name, label) for "full" level,
#              list of (field_name, label) for "name_target" level)
_TEMPLATES: dict[PerturbationType, tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = {
    PerturbationType.COMPOUND: (
        "Chemical perturbation: {info.pert_iname}.",
        [("target_list", "Target"), ("moa", "Function"), ("smiles", "SMILES")],
        [("target_list", "Target")],
    ),
    PerturbationType.CRISPR: (
        "CRISPR knockout of {info.gene}.",
        [("protein_name", "Protein"), ("moa", "Function"), ("go_terms", "GO terms")],
        [("protein_name", "Protein")],
    ),
    PerturbationType.ORF: (
        "ORF overexpression of {info.gene}.",
        [("protein_name", "Protein"), ("moa", "Function")],
        [("protein_name", "Protein")],
    ),
}


def generate_text(info: PerturbationInfo, level: str = "full") -> str:
    """Generate text description at the specified granularity.

    Args:
        info: Perturbation metadata.
        level: One of ``"name_only"``, ``"name_target"``, ``"full"``.

    Returns:
        Structured text description string.
    """
    # Controls always get the same text regardless of level
    if info.pert_type == PerturbationType.NEGCON:
        label = info.control_type or "DMSO"
        return f"Negative control ({label})."
    if info.pert_type == PerturbationType.POSCON:
        label = info.control_type or "positive"
        return f"Positive control ({label})."

    template = _TEMPLATES.get(info.pert_type)
    if template is None:
        return f"Unknown perturbation: {info.broad_sample}."

    prefix, full_fields, target_fields = template
    parts = [prefix.format(info=info)]

    if level == "name_only":
        return parts[0]

    extras = full_fields if level == "full" else target_fields
    for field, label in extras:
        value = getattr(info, field)
        if value:
            parts.append(f"{label}: {value}.")

    return " ".join(parts)


def well_from_row_col(row: int, col: int) -> str:
    """Convert numeric (row, col) to well position string.

    384-well plate: rows 1-16 map to A-P, columns 1-24 are zero-padded.
    """
    if not (1 <= row <= 16):
        raise ValueError(f"Row must be 1-16, got {row}")
    if not (1 <= col <= 24):
        raise ValueError(f"Column must be 1-24, got {col}")
    return f"{chr(64 + row)}{col:02d}"


def row_col_from_well(well: str) -> tuple[int, int]:
    """Convert well position string to numeric (row, col)."""
    letter = well[0].upper()
    col = int(well[1:])
    row = ord(letter) - 64
    return row, col


def is_control_or_empty(info: PerturbationInfo) -> bool:
    """Check if a perturbation is a control or has no broad_sample identifier."""
    return info.pert_type in {PerturbationType.NEGCON, PerturbationType.POSCON} or (
        not info.broad_sample
    )


def extract_plate_barcode(plate_dir_name: str) -> str:
    """Extract plate barcode from a plate directory name.

    Plate directories have format ``BR00116991__2020-11-05T19_51_35-Measurement1``.
    The barcode is the prefix before ``__``.
    """
    return plate_dir_name.split("__")[0]
