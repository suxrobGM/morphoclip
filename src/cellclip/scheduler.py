"""Sequential experiment scheduler for CellCLIP ChemBERTa sweeps.

Config dataclasses live in :mod:`cellclip.scheduler_spec`, stage builders in
:mod:`cellclip.scheduler_planning`, ranking in :mod:`cellclip.scheduler_ranking`,
and manifest/CSV I/O in :mod:`cellclip.scheduler_io`. This coordinator keeps
``PROJECT_ROOT`` and every function that reads it (path builders, the runner,
overlay-config writer, job runner, and ``run_schedule``) — tests monkeypatch
``cellclip.scheduler.PROJECT_ROOT``, which is resolved in this module's namespace
at call time.
"""

import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

from cellclip.scheduler_io import _read_json, _write_csv, append_manifest, load_manifest
from cellclip.scheduler_planning import (
    build_stage1_jobs,
    build_stage2_jobs,
    build_stage3_jobs,
)
from cellclip.scheduler_ranking import _leaderboard_rows, rank_records
from cellclip.scheduler_report import write_final_report
from cellclip.scheduler_spec import (
    BENCHMARK_COMPARE_KEYS,
    TUNING_MODE_OVERRIDES,
    CandidateSpec,
    Runner,
    ScheduleSpec,
    StageBudget,
    StageJob,
    _now_iso,
    _read_yaml,
)

__all__ = [
    "PROJECT_ROOT",
    "ScheduleSpec",
    "load_schedule_spec",
    "build_stage1_jobs",
    "build_stage2_jobs",
    "rank_records",
    "load_manifest",
    "run_schedule",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


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
