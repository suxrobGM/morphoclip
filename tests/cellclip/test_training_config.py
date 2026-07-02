"""Tests for CellCLIP training config loading (extends inheritance)."""

from pathlib import Path

import pytest

from cellclip.training.config import load_training_config


def test_load_training_config_supports_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "\n".join(
            [
                "dataset:",
                '  feature_root: "data/features_cellclip_base"',
                "  batch_size: 16",
                "optimization:",
                "  lr: 5.0e-4",
                "  warmup_steps: 100",
                "runtime:",
                '  run_name: "base_run"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "\n".join(
            [
                'extends: "base.yaml"',
                "dataset:",
                "  batch_size: 64",
                "optimization:",
                "  warmup_steps: 1000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_training_config(child)

    assert config.dataset.feature_root == "data/features_cellclip_base"
    assert config.dataset.batch_size == 64
    assert config.optimization.lr == pytest.approx(5.0e-4)
    assert config.optimization.warmup_steps == 1000
    assert config.runtime.run_name == "base_run"
