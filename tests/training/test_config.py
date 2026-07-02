"""Tests for MorphoCLIP training config."""

from pathlib import Path

import yaml

from morphoclip.training.config import (
    MorphoCLIPTrainingConfig,
    load_training_config,
)


class TestMorphoCLIPTrainingConfig:
    """Tests for config creation and serialization."""

    def test_default_creation(self) -> None:
        config = MorphoCLIPTrainingConfig()
        assert config.model.embed_dim == 1024
        assert config.model.output_dim == 512
        assert config.optimization.loss_type == "infonce"
        assert config.optimization.lr == 3.0e-4
        assert config.runtime.device == "auto"
        assert config.dataset.batch_size == 32

    def test_to_dict_roundtrip(self) -> None:
        config = MorphoCLIPTrainingConfig()
        d = config.to_dict()
        assert d["model"]["embed_dim"] == 1024
        assert d["optimization"]["loss_type"] == "infonce"
        assert d["runtime"]["amp"] is True

    def test_yaml_load(self, tmp_path: Path) -> None:
        config_data = {
            "dataset": {"batch_size": 16},
            "model": {"ccf_layers": 4},
            "optimization": {"lr": 1.0e-3, "epochs": 10},
            "runtime": {"seed": 123},
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_training_config(config_path)
        assert config.dataset.batch_size == 16
        assert config.model.ccf_layers == 4
        assert config.optimization.lr == 1.0e-3
        assert config.optimization.epochs == 10
        assert config.runtime.seed == 123
        # Defaults should still be in place for unset fields
        assert config.model.embed_dim == 1024
        assert config.optimization.loss_type == "infonce"

    def test_yaml_extends(self, tmp_path: Path) -> None:
        base = {"model": {"embed_dim": 512, "ccf_layers": 2}}
        child = {
            "extends": "base.yaml",
            "model": {"ccf_layers": 4},
        }
        (tmp_path / "base.yaml").write_text(yaml.dump(base))
        (tmp_path / "child.yaml").write_text(yaml.dump(child))

        config = load_training_config(tmp_path / "child.yaml")
        assert config.model.embed_dim == 512  # from base
        assert config.model.ccf_layers == 4  # overridden by child
