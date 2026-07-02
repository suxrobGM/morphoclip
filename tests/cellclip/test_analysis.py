"""Tests for reusable CellCLIP run analysis helpers."""

from pathlib import Path

import pytest
import torch

from cellclip.training.analysis import (
    build_comparison,
    compute_grouped_retrieval_metrics,
    compute_pca_stats,
    compute_perturbation_retrieval_metrics,
    compute_split_pca_stats,
    write_analysis_outputs,
)


def test_compute_grouped_retrieval_metrics_counts_equivalent_groups() -> None:
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    text_features = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    metrics = compute_grouped_retrieval_metrics(
        image_features,
        text_features,
        prompts=["same", "same"],
        broad_samples=["BRD-1", "BRD-1"],
    )

    assert metrics["exact_R@1"] == pytest.approx(0.0)
    assert metrics["prompt_R@1"] == pytest.approx(1.0)
    assert metrics["broad_sample_R@1"] == pytest.approx(1.0)


def test_compute_pca_stats_detects_rank_one_features() -> None:
    features = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])

    stats = compute_pca_stats(features)

    assert stats["top1_fraction"] == pytest.approx(1.0)
    assert stats["top10_fraction"] == pytest.approx(1.0)
    assert stats["mean_norm"] > 0.0


def test_compute_perturbation_retrieval_metrics_splits_by_perturbation_type() -> None:
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.3, 0.7]])
    text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]])

    metrics = compute_perturbation_retrieval_metrics(
        image_features,
        text_features,
        prompts=["c1", "c2", "g1", "g2"],
        broad_samples=["B1", "B2", "G1", "G2"],
        pert_types=["compound", "compound", "crispr", "crispr"],
    )

    assert metrics["compound"]["exact_R@1"] == pytest.approx(1.0)
    assert metrics["crispr"]["exact_R@1"] == pytest.approx(1.0)


def test_compute_split_pca_stats_handles_compound_and_noncompound_masks() -> None:
    image_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    text_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    stats = compute_split_pca_stats(
        image_features,
        text_features,
        pert_types=["compound", "compound", "crispr"],
    )

    assert stats["compound"]["image"]["samples"] == 2
    assert stats["non_compound"]["text"]["samples"] == 1


def test_write_analysis_outputs_writes_summary_and_report(tmp_path: Path) -> None:
    primary = {
        "run_dir": "/tmp/run_a",
        "checkpoint_path": "/tmp/run_a/checkpoints/best.pt",
        "metrics_history": [],
        "final_metrics": {"eval_loss": 1.23},
        "eval_retrieval": {"exact_R@1": 0.25},
        "compound_eval_retrieval": {"exact_R@1": 0.5},
        "perturbation_retrieval": {"compound": {"exact_R@1": 0.5}},
        "duplicate_stats": {"train": {"samples": 4}, "eval": {"samples": 2}},
        "has_smiles_fraction": 0.5,
        "fusion_diagnostics": {"fusion_delta_norm": 0.3},
        "image_pca": {"top1_fraction": 0.5},
        "text_pca": {"top1_fraction": 0.7},
        "split_pca": {"compound": {"image": {"top1_fraction": 0.5}}},
    }
    secondary = {
        **primary,
        "run_dir": "/tmp/run_b",
        "eval_retrieval": {"exact_R@1": 0.1},
        "compound_eval_retrieval": {"exact_R@1": 0.3},
        "image_pca": {"top1_fraction": 0.8},
        "text_pca": {"top1_fraction": 0.9},
        "fusion_diagnostics": {"fusion_delta_norm": 0.1},
    }

    summary_path, report_path = write_analysis_outputs(
        tmp_path,
        primary,
        secondary=secondary,
        comparison=build_comparison(primary, secondary),
    )

    assert summary_path.exists()
    assert report_path.exists()
    assert "CellCLIP Run Analysis" in report_path.read_text(encoding="utf-8")
