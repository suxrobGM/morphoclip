"""Tests for MorphoCLIP training config loading."""

from pathlib import Path

import pytest
import yaml

from morphoclip.training.config import (
    load_training_config,
    training_config_from_dict,
)

TRAIN_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "train"


class TestShippedConfigs:
    def test_base_yaml_still_loads_with_current_behavior(self) -> None:
        """base.yaml must stay behavior-identical: every new knob stays off."""
        config = load_training_config(TRAIN_CONFIGS / "base.yaml")
        assert config.optimization.loss_type == "cwcl"
        assert config.model.aggregator == "ccf-mean"
        assert config.dataset.batch_sampler == "random"
        assert config.dataset.replicates_per_group == 2
        assert config.optimization.target_weight == 0.0
        assert config.optimization.replicate_weight == 0.0
        assert config.optimization.replicate_temperature is None
        assert config.runtime.early_stop_patience is None

    def test_mean_pool_yaml_uses_new_aggregator_field(self) -> None:
        config = load_training_config(TRAIN_CONFIGS / "mean_pool.yaml")
        assert config.model.aggregator == "meanpool-mean"

    def test_to_dict_round_trips_through_the_schema(self) -> None:
        """Checkpoints store `to_dict()`. Resume rebuilds from it, so it has to validate."""
        config = load_training_config(TRAIN_CONFIGS / "base.yaml")
        assert training_config_from_dict(config.to_dict()) == config


class TestLoadTrainingConfig:
    def test_yaml_load_keeps_defaults_for_unset_fields(self, tmp_path: Path) -> None:
        config_path = tmp_path / "test_config.yaml"
        config_path.write_text(
            yaml.dump(
                {
                    "dataset": {"batch_size": 16},
                    "model": {"ccf_layers": 4},
                    "optimization": {"lr": 1.0e-3, "epochs": 10},
                    "runtime": {"seed": 123},
                }
            )
        )

        config = load_training_config(config_path)
        assert config.dataset.batch_size == 16
        assert config.model.ccf_layers == 4
        assert config.optimization.lr == 1.0e-3
        assert config.optimization.epochs == 10
        assert config.runtime.seed == 123
        assert config.model.output_dim == 512
        assert config.optimization.loss_type == "infonce"

    def test_yaml_extends(self, tmp_path: Path) -> None:
        (tmp_path / "base.yaml").write_text(
            yaml.dump({"model": {"output_dim": 256, "ccf_layers": 2}})
        )
        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({"extends": "base.yaml", "model": {"ccf_layers": 4}}))

        config = load_training_config(child)
        assert config.model.output_dim == 256
        assert config.model.ccf_layers == 4

    def test_dotted_overrides_apply_after_extends(self, tmp_path: Path) -> None:
        (tmp_path / "base.yaml").write_text(yaml.dump({"dataset": {"batch_size": 32}}))
        child = tmp_path / "child.yaml"
        child.write_text(yaml.dump({"extends": "base.yaml"}))

        config = load_training_config(
            child, overrides=["dataset.batch_size=8", "runtime.max_train_steps=3"]
        )
        assert config.dataset.batch_size == 8
        assert config.runtime.max_train_steps == 3

    @pytest.mark.parametrize("override", ["dataset.batch_size", "batch_size=8"])
    def test_malformed_dotted_override_rejected(self, tmp_path: Path, override: str) -> None:
        config_path = tmp_path / "base.yaml"
        config_path.write_text(yaml.dump({}))

        with pytest.raises(ValueError, match="Malformed --set"):
            load_training_config(config_path, overrides=[override])

    def test_unknown_model_key_rejected(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad_model.yaml"
        config_path.write_text(yaml.dump({"model": {"channel_aggregation": "ccf"}}))

        with pytest.raises(ValueError, match=r"model\.channel_aggregation"):
            load_training_config(config_path)


def test_training_config_from_dict_rejects_unknown_keys() -> None:
    """`betas` is a CellCLIP key. Loading it here must name the offending path."""
    with pytest.raises(ValueError, match=r"optimization\.betas"):
        training_config_from_dict({"optimization": {"betas": [0.9, 0.999]}})
