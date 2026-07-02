"""Sequential experiment scheduler for CellCLIP ChemBERTa sweeps."""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from cellclip.scheduler_report import write_final_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return payload


def load_schedule_spec(path: str | Path) -> ScheduleSpec:
    """Load a scheduler spec from YAML."""
    raw = _read_yaml(_resolve_project_path(path))
    compare_dirs = {
        key: _resolve_project_path(raw["compare_full_benchmark_dirs"][key])
        for key in BENCHMARK_COMPARE_KEYS
    }
    budgets = {
        name: StageBudget(
            name=name,
            max_train_steps=payload.get("max_train_steps"),
            max_eval_wells=payload.get("max_eval_wells"),
            promote_top=payload.get("promote_top"),
        )
        for name, payload in raw["stage_budgets"].items()
    }
    candidates = [
        CandidateSpec(
            candidate_id=item["id"],
            model=dict(item.get("model", {})),
            dataset=dict(item.get("dataset", {})),
        )
        for item in raw["stage1_candidates"]
    ]
    tuning_modes = [str(mode).strip().lower() for mode in raw["stage2_tuning_modes"]]
    invalid_modes = [mode for mode in tuning_modes if mode not in TUNING_MODE_OVERRIDES]
    if invalid_modes:
        raise ValueError(f"Unsupported tuning modes: {invalid_modes}")
    benchmark_timelines = tuple(
        str(item).strip().lower() for item in raw.get("benchmark_timelines", ["short", "long"])
    )
    if not benchmark_timelines:
        raise ValueError("benchmark_timelines must contain at least one timeline")
    invalid_timelines = [item for item in benchmark_timelines if item not in {"short", "long"}]
    if invalid_timelines:
        raise ValueError(f"Unsupported benchmark timelines: {invalid_timelines}")
    return ScheduleSpec(
        schedule_name=str(raw["schedule_name"]).strip(),
        base_config=_resolve_project_path(raw["base_config"]),
        compare_full_benchmark_dirs=compare_dirs,
        stage_budgets=budgets,
        stage1_candidates=candidates,
        stage2_tuning_modes=tuning_modes,
        benchmark_timelines=benchmark_timelines,
    )


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


def _score_tuple(record: dict[str, Any]) -> tuple[float, float, float, float, float]:
    summary = record.get("analysis_summary", {})
    primary = summary.get("primary", summary)
    compound = primary.get("compound_eval_retrieval", {})
    overall = primary.get("eval_retrieval", {})
    metrics = primary.get("final_metrics", {})
    split_pca = primary.get("split_pca", {}).get("compound", {})
    image_pca = split_pca.get("image", {})
    text_pca = split_pca.get("text", {})
    return (
        -float(compound.get("broad_sample_R@1", 0.0)),
        -float(overall.get("broad_sample_R@1", 0.0)),
        -float(metrics.get("text_to_image_R@10", 0.0)),
        float(image_pca.get("top1_fraction", 1.0)),
        float(text_pca.get("top1_fraction", 1.0)),
    )


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort completed records using the fixed promotion order."""
    completed = [record for record in records if record.get("status") == "completed"]
    return sorted(completed, key=_score_tuple)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the latest manifest row for each stage/candidate pair."""
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["stage"], row["candidate_id"])
            records[key] = row
    return records


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    """Append one scheduler record to a JSONL manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def default_runner(command: list[str], log_path: Path) -> None:
    """Run one command and tee stdout/stderr into a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(shlex.quote(part) for part in command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _train_run_name(spec: ScheduleSpec, job: StageJob) -> str:
    return f"{spec.schedule_name}__{job.candidate_id}__{job.stage}"


def _schedule_dir(spec: ScheduleSpec) -> Path:
    return PROJECT_ROOT / "output" / "sweeps" / spec.schedule_name


def _generated_config_path(spec: ScheduleSpec, job: StageJob) -> Path:
    return _schedule_dir(spec) / "configs" / f"{job.candidate_id}__{job.stage}.yaml"


def _train_run_dir(run_name: str) -> Path:
    return PROJECT_ROOT / "output" / "train_runs" / run_name


def _benchmark_dirs(spec: ScheduleSpec, job: StageJob) -> tuple[Path, Path]:
    if spec.benchmark_timelines == ("short", "long"):
        label = "full"
    elif len(spec.benchmark_timelines) == 1:
        label = spec.benchmark_timelines[0]
    else:
        label = "_".join(spec.benchmark_timelines)
    suffix = f"{spec.schedule_name}__{job.candidate_id}__{label}"
    profiles_dir = PROJECT_ROOT / "data" / "profiles_scheduler" / suffix
    benchmark_dir = PROJECT_ROOT / "output" / f"benchmark_{suffix}"
    return profiles_dir, benchmark_dir


def _write_overlay_config(
    spec: ScheduleSpec,
    job: StageJob,
    budget: StageBudget,
) -> Path:
    payload: dict[str, Any] = {"extends": str(spec.base_config)}
    if job.model_overrides:
        payload["model"] = dict(job.model_overrides)
    dataset_overrides: dict[str, Any] = {}
    runtime_overrides: dict[str, Any] = {}
    dataset_overrides.update(job.dataset_overrides)
    if budget.max_eval_wells is not None:
        dataset_overrides["max_eval_wells"] = budget.max_eval_wells
    if budget.max_train_steps is not None:
        runtime_overrides["max_train_steps"] = budget.max_train_steps
    if dataset_overrides:
        payload["dataset"] = dataset_overrides
    if runtime_overrides:
        payload["runtime"] = runtime_overrides
    path = _generated_config_path(spec, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_job(
    spec: ScheduleSpec,
    job: StageJob,
    budget: StageBudget,
    runner: Runner,
) -> dict[str, Any]:
    run_name = _train_run_name(spec, job)
    run_dir = _train_run_dir(run_name)
    config_path = _write_overlay_config(spec, job, budget)
    record: dict[str, Any] = {
        "schedule_name": spec.schedule_name,
        "stage": job.stage,
        "candidate_id": job.candidate_id,
        "family_id": job.family_id,
        "model_overrides": dict(job.model_overrides),
        "dataset_overrides": dict(job.dataset_overrides),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "status": "running",
        "started_at": _now_iso(),
    }
    try:
        runner(
            [
                "uv",
                "run",
                "python",
                "scripts/cellclip/train_cellclip.py",
                "--config",
                str(config_path),
                "--run-name",
                run_name,
            ],
            run_dir / "train.log",
        )
        analysis_dir = run_dir / "analysis_scheduler"
        analyze_cmd = [
            "uv",
            "run",
            "python",
            "scripts/cellclip/analyze_training_run.py",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(analysis_dir),
        ]
        if budget.max_eval_wells is not None:
            analyze_cmd.extend(["--max-eval-wells", str(budget.max_eval_wells)])
        runner(analyze_cmd, analysis_dir / "analysis.log")
        record["analysis_dir"] = str(analysis_dir)
        record["analysis_summary_path"] = str(analysis_dir / "summary.json")
        record["analysis_summary"] = _read_json(analysis_dir / "summary.json")
        if job.stage == "stage3":
            profiles_dir, benchmark_dir = _benchmark_dirs(spec, job)
            export_cmd = [
                "uv",
                "run",
                "python",
                "scripts/cellclip/export_cellclip_profiles.py",
                "--config",
                "configs/benchmark.yml",
                "--ckpt-path",
                str(run_dir / "checkpoints" / "best.pt"),
                "--output-profiles-root",
                str(profiles_dir),
                "--timelines",
                *spec.benchmark_timelines,
            ]
            runner(export_cmd, benchmark_dir / "export.log")
            benchmark_cmd = [
                "uv",
                "run",
                "python",
                "scripts/benchmark/benchmark_stable.py",
                "--config",
                "configs/benchmark.yml",
                "--profiles-dir",
                str(profiles_dir),
                "--output-dir",
                str(benchmark_dir),
                "--timelines",
                *spec.benchmark_timelines,
            ]
            runner(benchmark_cmd, benchmark_dir / "benchmark.log")
            record["profiles_dir"] = str(profiles_dir)
            record["benchmark_dir"] = str(benchmark_dir)
        record["status"] = "completed"
    except subprocess.CalledProcessError as exc:
        record["status"] = "failed"
        record["error"] = f"command exited with {exc.returncode}: {' '.join(exc.cmd)}"
    record["finished_at"] = _now_iso()
    return record


def _leaderboard_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, record in enumerate(rank_records(records), start=1):
        primary = record["analysis_summary"]["primary"]
        rows.append(
            {
                "rank": rank,
                "candidate_id": record["candidate_id"],
                "family_id": record["family_id"],
                "stage": record["stage"],
                "status": record["status"],
                "compound_broad_sample_R@1": primary["compound_eval_retrieval"].get(
                    "broad_sample_R@1", 0.0
                ),
                "overall_broad_sample_R@1": primary["eval_retrieval"].get("broad_sample_R@1", 0.0),
                "text_to_image_R@10": primary["final_metrics"].get("text_to_image_R@10", 0.0),
                "compound_image_top1_fraction": primary["split_pca"]["compound"]["image"].get(
                    "top1_fraction", 1.0
                ),
                "compound_text_top1_fraction": primary["split_pca"]["compound"]["text"].get(
                    "top1_fraction", 1.0
                ),
                "run_dir": record["run_dir"],
                "analysis_summary_path": record["analysis_summary_path"],
            }
        )
    return rows


def print_dry_run(spec: ScheduleSpec) -> None:
    """Print the fully expanded scheduler plan without running anything."""
    stage1 = build_stage1_jobs(spec)
    print(f"Schedule: {spec.schedule_name}")
    print("Stage 1:")
    for job in stage1:
        print(
            f"  - {job.candidate_id}: model={job.model_overrides} dataset={job.dataset_overrides}"
        )
    print("Stage 2 template:")
    print(f"  - Top {spec.stage_budgets['stage1'].promote_top} Stage 1 families expand into:")
    for tuning_mode in spec.stage2_tuning_modes:
        print(f"    * <family>__{tuning_mode}: {TUNING_MODE_OVERRIDES[tuning_mode]}")
    print(
        "Stage 3 finalists: top "
        f"{spec.stage_budgets['stage2'].promote_top} Stage 2 candidates run full training + "
        f"benchmark timelines {list(spec.benchmark_timelines)}"
    )


def run_schedule(
    spec: ScheduleSpec, *, dry_run: bool = False, resume: bool = False, runner: Runner | None = None
) -> Path:
    """Execute or preview the full ChemBERTa schedule."""
    if dry_run:
        print_dry_run(spec)
        return _schedule_dir(spec)
    run_command = runner or default_runner
    schedule_dir = _schedule_dir(spec)
    manifest_path = schedule_dir / "manifest.jsonl"
    manifest = load_manifest(manifest_path) if resume else {}

    def process_stage(stage_name: str, jobs: list[StageJob]) -> list[dict[str, Any]]:
        budget = spec.stage_budgets[stage_name]
        completed: list[dict[str, Any]] = []
        for job in jobs:
            key = (stage_name, job.candidate_id)
            existing = manifest.get(key)
            if existing and existing.get("status") == "completed":
                completed.append(existing)
                continue
            record = _run_job(spec, job, budget, run_command)
            append_manifest(manifest_path, record)
            manifest[key] = record
            completed.append(record)
        _write_csv(schedule_dir / f"leaderboard_{stage_name}.csv", _leaderboard_rows(completed))
        return completed

    stage1_records = process_stage("stage1", build_stage1_jobs(spec))
    promoted_stage1 = [
        record["family_id"]
        for record in rank_records(stage1_records)[: spec.stage_budgets["stage1"].promote_top]
    ]
    stage2_records = process_stage("stage2", build_stage2_jobs(spec, promoted_stage1))
    promoted_stage2 = rank_records(stage2_records)[: spec.stage_budgets["stage2"].promote_top]
    stage3_records = process_stage("stage3", build_stage3_jobs(promoted_stage2))
    write_final_report(spec, schedule_dir, stage3_records)
    return schedule_dir
