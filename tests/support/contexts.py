"""Redirect the reference metadata that :mod:`morphoclip.splits.contexts` reads.

The resolvers walk a list of candidate paths, so a test that redirects only some
of them silently reads the repo's real ``data/reference/cpjump1/`` copy instead
of failing. Naming the candidates once here keeps that list in one place.
"""

from pathlib import Path

import pytest

import morphoclip.splits.contexts as split_contexts

OFFICIAL_CSV_HEADER = (
    "Metadata_Plate,Metadata_Well,Metadata_broad_sample,Metadata_target,"
    "Metadata_cell_line,Metadata_experiment_type,Metadata_timepoint,"
    "Metadata_timepoint_code,Metadata_target_is_across,Metadata_target_radix"
)

_EXPERIMENT_TSV_CONSTANTS = ("METADATA_PATH", "FALLBACK_METADATA_PATH", "REFERENCE_METADATA_PATH")


def hide_experiment_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every experiment-TSV candidate at a missing file.

    Keeps the plate-condition fallback out of tests that only exercise the
    official CSV.
    """
    for constant in _EXPERIMENT_TSV_CONSTANTS:
        monkeypatch.setattr(split_contexts, constant, tmp_path / "missing.tsv")


def write_official_split_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rows: tuple[str, ...],
    *,
    hide_experiment_tsv: bool = True,
) -> Path:
    """Write an official-split CSV stand-in and point the resolver at it.

    Args:
        monkeypatch: Patcher for the module constants.
        tmp_path: Directory to write into.
        rows: Data rows, without the header.
        hide_experiment_tsv: Also redirect the experiment-TSV candidates, so the
            condition fallback cannot read the repo's real copy.

    Returns:
        Path to the written CSV.
    """
    path = tmp_path / "cpjump1_metadata.csv"
    path.write_text("\n".join((OFFICIAL_CSV_HEADER, *rows)) + "\n", encoding="utf-8")
    monkeypatch.setattr(split_contexts, "OFFICIAL_SPLIT_METADATA_PATH", path)
    if hide_experiment_tsv:
        hide_experiment_metadata(monkeypatch, tmp_path)
    return path
