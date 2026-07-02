"""Config dataclasses, constants, and YAML helpers for the CellCLIP sweep scheduler."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

BENCHMARK_COMPARE_KEYS = ("baseline", "pretrained_clip")
TUNING_MODE_OVERRIDES = {
    "frozen": {"freeze_chemberta": True, "chemberta_tune_layers": 0},
    "top2": {"freeze_chemberta": False, "chemberta_tune_layers": 2},
    "full": {"freeze_chemberta": False, "chemberta_tune_layers": 0},
}


@dataclass(frozen=True)
class StageBudget:
    """Budget and promotion settings for one scheduler stage."""

    name: str
    max_train_steps: int | None
    max_eval_wells: int | None
    promote_top: int | None


@dataclass(frozen=True)
class CandidateSpec:
    """One ChemBERTa candidate family."""

    candidate_id: str
    model: dict[str, Any]
    dataset: dict[str, Any]


@dataclass(frozen=True)
class ScheduleSpec:
    """Resolved sweep configuration."""

    schedule_name: str
    base_config: Path
    compare_full_benchmark_dirs: dict[str, Path]
    stage_budgets: dict[str, StageBudget]
    stage1_candidates: list[CandidateSpec]
    stage2_tuning_modes: list[str]
    benchmark_timelines: tuple[str, ...]


@dataclass(frozen=True)
class StageJob:
    """One runnable stage candidate."""

    stage: str
    candidate_id: str
    family_id: str
    model_overrides: dict[str, Any]
    dataset_overrides: dict[str, Any]


Runner = Callable[[list[str], Path], None]


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return payload
