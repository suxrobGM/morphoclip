"""Stage job builders for the CellCLIP sweep scheduler."""

from __future__ import annotations

from typing import Any

from cellclip.scheduler_spec import TUNING_MODE_OVERRIDES, ScheduleSpec, StageJob


def build_stage1_jobs(spec: ScheduleSpec) -> list[StageJob]:
    """Return the fixed Stage 1 ChemBERTa candidate set."""
    return [
        StageJob(
            stage="stage1",
            candidate_id=candidate.candidate_id,
            family_id=candidate.candidate_id,
            model_overrides=dict(candidate.model),
            dataset_overrides=dict(candidate.dataset),
        )
        for candidate in spec.stage1_candidates
    ]


def build_stage2_jobs(spec: ScheduleSpec, promoted_families: list[str]) -> list[StageJob]:
    """Expand promoted Stage 1 families across tuning modes."""
    family_map = {candidate.candidate_id: candidate for candidate in spec.stage1_candidates}
    jobs: list[StageJob] = []
    for family_id in promoted_families:
        base_candidate = family_map[family_id]
        for tuning_mode in spec.stage2_tuning_modes:
            overrides = dict(base_candidate.model)
            overrides.update(TUNING_MODE_OVERRIDES[tuning_mode])
            jobs.append(
                StageJob(
                    stage="stage2",
                    candidate_id=f"{family_id}__{tuning_mode}",
                    family_id=family_id,
                    model_overrides=overrides,
                    dataset_overrides=dict(base_candidate.dataset),
                )
            )
    return jobs


def build_stage3_jobs(stage2_records: list[dict[str, Any]]) -> list[StageJob]:
    """Promote selected Stage 2 records into full runs."""
    jobs: list[StageJob] = []
    for record in stage2_records:
        jobs.append(
            StageJob(
                stage="stage3",
                candidate_id=record["candidate_id"],
                family_id=record["family_id"],
                model_overrides=dict(record["model_overrides"]),
                dataset_overrides=dict(record.get("dataset_overrides", {})),
            )
        )
    return jobs
