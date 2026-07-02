"""Tests for the CellCLIP ChemBERTa scheduler."""

import csv
import json
from pathlib import Path

import pytest

from cellclip.scheduler import (
    build_stage1_jobs,
    build_stage2_jobs,
    load_manifest,
    load_schedule_spec,
    rank_records,
    run_schedule,
)


def _posix(p: Path) -> str:
    """Convert a Path to a POSIX string safe for YAML double-quoted values."""
    return p.as_posix()


def _write_summary_csv(path: Path, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "gene_compound_matching_summary.csv":
        path.write_text(
            "\n".join(
                [
                    "Modality1,Modality2,A549,U2OS",
                    "compound,crispr,0.0,0.0",
                    "compound,orf,0.0,0.0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return
    path.write_text(
        "\n".join(
            [
                "Modality,time,A549,U2OS",
                "compound,short,0.1,0.2",
                "compound,long,0.3,0.4",
                "crispr,short,0.5,0.6",
                "crispr,long,0.7,0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_compare_tables(root: Path) -> None:
    for filename in (
        "replicability_summary.csv",
        "matching_summary.csv",
        "gene_compound_matching_summary.csv",
    ):
        _write_summary_csv(root / "tables" / filename, filename)


def _analysis_payload(score: float) -> dict:
    return {
        "primary": {
            "compound_eval_retrieval": {"broad_sample_R@1": score},
            "eval_retrieval": {"broad_sample_R@1": score / 2},
            "final_metrics": {"text_to_image_R@10": score / 4},
            "split_pca": {
                "compound": {
                    "image": {"top1_fraction": 1.0 - score / 10},
                    "text": {"top1_fraction": 1.0 - score / 20},
                }
            },
        }
    }


def _candidate_score(candidate_id: str) -> float:
    base_scores = {
        "film_remove_mean": 0.10,
        "film_keep_mean": 0.35,
        "film_keep_cls": 0.30,
        "residual_keep_mean": 0.60,
        "residual_keep_cls": 0.55,
        "concat_keep_mean": 0.50,
        "concat_keep_cls": 0.45,
        "same_pert_3": 0.25,
        "same_pert_7": 0.35,
        "same_pert_11": 0.45,
    }
    if "__" not in candidate_id:
        return base_scores.get(candidate_id, 0.20)
    family_id, tuning_mode = candidate_id.split("__", 1)
    tuning_bonus = {"frozen": 0.00, "top2": 0.05, "full": 0.10}[tuning_mode]
    return base_scores.get(family_id, 0.20) + tuning_bonus


class FakeRunner:
    """Mock scheduler runner that materializes expected stage outputs."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], log_path: Path) -> None:
        self.calls.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        script = next((part for part in command if part.endswith(".py")), "")
        if script.endswith("train_cellclip.py"):
            run_name = command[command.index("--run-name") + 1]
            run_dir = self.project_root / "output" / "train_runs" / run_name
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints" / "best.pt").write_text("pt\n", encoding="utf-8")
        elif script.endswith("analyze_training_run.py"):
            run_dir = Path(command[command.index("--run-dir") + 1])
            candidate_id = run_dir.name.split("__")[1]
            output_dir = Path(command[command.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = _analysis_payload(_candidate_score(candidate_id))
            (output_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        elif script.endswith("benchmark_stable.py"):
            output_dir = Path(command[command.index("--output-dir") + 1])
            for filename in (
                "replicability_summary.csv",
                "matching_summary.csv",
                "gene_compound_matching_summary.csv",
            ):
                _write_summary_csv(output_dir / "tables" / filename, filename)
        elif script.endswith("export_cellclip_profiles.py"):
            profiles_dir = Path(command[command.index("--output-profiles-root") + 1])
            profiles_dir.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def scheduler_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from cellclip import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PROJECT_ROOT", tmp_path)
    spec_dir = tmp_path / "configs" / "cellclip" / "schedules"
    spec_dir.mkdir(parents=True, exist_ok=True)
    compare_base = tmp_path / "output" / "benchmark_full_baseline"
    compare_clip = tmp_path / "output" / "benchmark_full_cellclip_hf"
    _write_compare_tables(compare_base)
    _write_compare_tables(compare_clip)
    spec_path = spec_dir / "chemberta_full_benchmark.yaml"
    spec_path.write_text(
        "\n".join(
            [
                'schedule_name: "chemberta_full_benchmark"',
                f'base_config: "{_posix(tmp_path / "base.yaml")}"',
                "compare_full_benchmark_dirs:",
                f'  baseline: "{_posix(compare_base)}"',
                f'  pretrained_clip: "{_posix(compare_clip)}"',
                "stage_budgets:",
                "  stage1:",
                "    max_train_steps: 1500",
                "    max_eval_wells: 1024",
                "    promote_top: 4",
                "  stage2:",
                "    max_train_steps: 4500",
                "    max_eval_wells: null",
                "    promote_top: 2",
                "  stage3:",
                "    max_train_steps: null",
                "    max_eval_wells: null",
                "    promote_top: null",
                "stage2_tuning_modes:",
                '  - "frozen"',
                '  - "top2"',
                '  - "full"',
                "stage1_candidates:",
                '  - id: "film_remove_mean"',
                "    model: {variant: chemberta_film, chem_fusion_type: film, "
                "chem_prompt_policy: remove_smiles, chemberta_pooling: mean, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
                '  - id: "film_keep_mean"',
                "    model: {variant: chemberta, chem_fusion_type: film, "
                "chem_prompt_policy: keep_smiles, chemberta_pooling: mean, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
                '  - id: "film_keep_cls"',
                "    model: {variant: chemberta, chem_fusion_type: film, "
                "chem_prompt_policy: keep_smiles, chemberta_pooling: cls, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
                '  - id: "residual_keep_mean"',
                "    model: {variant: chemberta, chem_fusion_type: residual_add, "
                "chem_prompt_policy: keep_smiles, chemberta_pooling: mean, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
                '  - id: "residual_keep_cls"',
                "    model: {variant: chemberta, chem_fusion_type: residual_add, "
                "chem_prompt_policy: keep_smiles, chemberta_pooling: cls, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
                '  - id: "concat_keep_mean"',
                "    model: {variant: chemberta, chem_fusion_type: concat_mlp, "
                "chem_prompt_policy: keep_smiles, chemberta_pooling: mean, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
                '  - id: "concat_keep_cls"',
                "    model: {variant: chemberta, chem_fusion_type: concat_mlp, "
                "chem_prompt_policy: keep_smiles, chemberta_pooling: cls, "
                "freeze_chemberta: true, chemberta_tune_layers: 0}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text("runtime: {}\n", encoding="utf-8")
    return load_schedule_spec(spec_path)


def test_load_schedule_spec_and_stage_expansion(scheduler_spec) -> None:
    stage1 = build_stage1_jobs(scheduler_spec)
    assert len(stage1) == 7
    assert stage1[0].dataset_overrides == {}
    stage2 = build_stage2_jobs(scheduler_spec, ["residual_keep_mean", "concat_keep_cls"])
    assert len(stage2) == 6
    assert stage2[0].candidate_id == "residual_keep_mean__frozen"
    assert stage2[-1].candidate_id == "concat_keep_cls__full"
    assert stage2[0].dataset_overrides == {}


def test_rank_records_uses_fixed_promotion_order() -> None:
    ranked = rank_records(
        [
            {
                "status": "completed",
                "analysis_summary": _analysis_payload(0.2),
            },
            {
                "status": "completed",
                "analysis_summary": _analysis_payload(0.5),
            },
        ]
    )
    assert (
        ranked[0]["analysis_summary"]["primary"]["compound_eval_retrieval"]["broad_sample_R@1"]
        == 0.5
    )


def test_run_schedule_executes_three_stage_funnel(
    scheduler_spec,
    tmp_path: Path,
) -> None:
    runner = FakeRunner(tmp_path)
    schedule_dir = run_schedule(scheduler_spec, runner=runner)

    manifest = load_manifest(schedule_dir / "manifest.jsonl")
    assert ("stage1", "film_remove_mean") in manifest
    assert ("stage2", "residual_keep_mean__full") in manifest
    stage3_records = [record for (stage, _), record in manifest.items() if stage == "stage3"]
    assert len(stage3_records) == 2
    assert all(record["status"] == "completed" for record in stage3_records)
    assert all("benchmark_dir" in record for record in stage3_records)
    assert (schedule_dir / "leaderboard_stage1.csv").exists()
    assert (schedule_dir / "leaderboard_stage2.csv").exists()
    assert (schedule_dir / "final_report.md").exists()
    assert any("benchmark_stable.py" in call for command in runner.calls for call in command)


def test_run_schedule_resume_skips_completed_work(
    scheduler_spec,
    tmp_path: Path,
) -> None:
    runner = FakeRunner(tmp_path)
    schedule_dir = run_schedule(scheduler_spec, runner=runner)

    resumed_runner = FakeRunner(tmp_path)
    resumed_dir = run_schedule(scheduler_spec, resume=True, runner=resumed_runner)

    assert resumed_dir == schedule_dir
    assert len(resumed_runner.calls) == 0
    with open(schedule_dir / "leaderboard_stage1.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["candidate_id"] == "residual_keep_mean"


def test_load_schedule_spec_accepts_dataset_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellclip import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PROJECT_ROOT", tmp_path)
    spec_path = tmp_path / "schedule.yaml"
    spec_path.write_text(
        "\n".join(
            [
                'schedule_name: "chemberta_aug"',
                f'base_config: "{_posix(tmp_path / "base.yaml")}"',
                "compare_full_benchmark_dirs:",
                f'  baseline: "{_posix(tmp_path / "baseline")}"',
                f'  pretrained_clip: "{_posix(tmp_path / "clip")}"',
                "stage_budgets:",
                "  stage1: {max_train_steps: 10, max_eval_wells: 8, promote_top: 1}",
                "  stage2: {max_train_steps: 20, max_eval_wells: null, promote_top: 1}",
                "  stage3: {max_train_steps: null, max_eval_wells: null, promote_top: null}",
                "stage2_tuning_modes: [full]",
                "stage1_candidates:",
                '  - id: "mixed"',
                "    model: {variant: chemberta}",
                "    dataset: {within_well_interp_sites: 1, same_pert_interp_sites: 1}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text("runtime: {}\n", encoding="utf-8")
    (tmp_path / "baseline").mkdir()
    (tmp_path / "clip").mkdir()

    spec = load_schedule_spec(spec_path)
    jobs = build_stage1_jobs(spec)

    assert jobs[0].candidate_id == "mixed"
    assert jobs[0].dataset_overrides == {
        "within_well_interp_sites": 1,
        "same_pert_interp_sites": 1,
    }


def test_load_schedule_spec_accepts_short_only_benchmark_timelines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellclip import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PROJECT_ROOT", tmp_path)
    spec_path = tmp_path / "schedule_short.yaml"
    spec_path.write_text(
        "\n".join(
            [
                'schedule_name: "chemberta_short_only"',
                f'base_config: "{_posix(tmp_path / "base.yaml")}"',
                "compare_full_benchmark_dirs:",
                f'  baseline: "{_posix(tmp_path / "baseline")}"',
                f'  pretrained_clip: "{_posix(tmp_path / "clip")}"',
                'benchmark_timelines: ["short"]',
                "stage_budgets:",
                "  stage1: {max_train_steps: 10, max_eval_wells: 8, promote_top: 1}",
                "  stage2: {max_train_steps: 20, max_eval_wells: null, promote_top: 1}",
                "  stage3: {max_train_steps: null, max_eval_wells: null, promote_top: null}",
                "stage2_tuning_modes: [full]",
                "stage1_candidates:",
                '  - id: "same_pert_3"',
                "    dataset: {train_max_sites_per_well: 9, same_pert_interp_sites: 3}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text("runtime: {}\n", encoding="utf-8")
    (tmp_path / "baseline").mkdir()
    (tmp_path / "clip").mkdir()

    spec = load_schedule_spec(spec_path)

    assert spec.benchmark_timelines == ("short",)


def test_run_schedule_uses_configured_short_only_benchmark_timelines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cellclip import scheduler as scheduler_module

    monkeypatch.setattr(scheduler_module, "PROJECT_ROOT", tmp_path)
    compare_base = tmp_path / "output" / "benchmark_full_baseline"
    compare_clip = tmp_path / "output" / "benchmark_full_cellclip_hf"
    _write_compare_tables(compare_base)
    _write_compare_tables(compare_clip)
    spec_path = tmp_path / "configs" / "cellclip" / "schedules" / "short_only.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        "\n".join(
            [
                'schedule_name: "chemberta_short_only"',
                f'base_config: "{_posix(tmp_path / "base.yaml")}"',
                "compare_full_benchmark_dirs:",
                f'  baseline: "{_posix(compare_base)}"',
                f'  pretrained_clip: "{_posix(compare_clip)}"',
                'benchmark_timelines: ["short"]',
                "stage_budgets:",
                "  stage1: {max_train_steps: 10, max_eval_wells: 8, promote_top: 1}",
                "  stage2: {max_train_steps: 20, max_eval_wells: null, promote_top: 1}",
                "  stage3: {max_train_steps: null, max_eval_wells: null, promote_top: null}",
                "stage2_tuning_modes: [full]",
                "stage1_candidates:",
                '  - id: "same_pert_3"',
                "    dataset: {train_max_sites_per_well: 9, same_pert_interp_sites: 3}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "base.yaml").write_text("runtime: {}\n", encoding="utf-8")

    spec = load_schedule_spec(spec_path)
    runner = FakeRunner(tmp_path)
    schedule_dir = run_schedule(spec, runner=runner)
    manifest = load_manifest(schedule_dir / "manifest.jsonl")
    stage3 = manifest[("stage3", "same_pert_3__full")]

    assert stage3["benchmark_dir"].endswith("__short")
    stage3_calls = [
        cmd
        for cmd in runner.calls
        if any(
            part.endswith(("export_cellclip_profiles.py", "benchmark_stable.py")) for part in cmd
        )
    ]
    assert len(stage3_calls) == 2
    for command in stage3_calls:
        index = command.index("--timelines")
        assert command[index + 1 :] == ["short"]
