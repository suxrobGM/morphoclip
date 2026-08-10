"""Tests for MorphoCLIP training config."""

from pathlib import Path

import pytest
import yaml

from morphoclip.training.config import (
    MorphoCLIPTrainingConfig,
    load_training_config,
    training_config_from_dict,
)


class TestMorphoCLIPTrainingConfig:
    """Tests for config creation and serialization."""

    def test_training_signal_fields_default_off(self) -> None:
        """New features must default to current behavior."""
        config = MorphoCLIPTrainingConfig()
        assert config.dataset.batch_sampler == "random"
        assert config.dataset.replicates_per_group == 2
        assert config.optimization.target_weight == 0.0
        assert config.optimization.replicate_weight == 0.0
        assert config.optimization.replicate_temperature is None
        assert config.runtime.early_stop_patience is None

    def test_base_yaml_still_loads_with_current_behavior(self) -> None:
        base_yaml = Path("configs/train/base.yaml")
        if not base_yaml.exists():
            pytest.skip("configs/train/base.yaml not available")
        config = load_training_config(base_yaml)
        assert config.optimization.loss_type == "cwcl"
        # base.yaml must stay behavior-identical: every new knob stays off.
        assert config.model.aggregator == "ccf-mean"
        assert config.dataset.batch_sampler == "random"
        assert config.optimization.target_weight == 0.0
        assert config.optimization.replicate_weight == 0.0
        assert config.optimization.replicate_temperature is None
        assert config.runtime.early_stop_patience is None

    def test_mean_pool_yaml_uses_new_aggregator_field(self) -> None:
        mean_pool_yaml = Path("configs/train/mean_pool.yaml")
        if not mean_pool_yaml.exists():
            pytest.skip("configs/train/mean_pool.yaml not available")
        config = load_training_config(mean_pool_yaml)
        assert config.model.aggregator == "meanpool-mean"

    def test_unknown_model_key_rejected(self, tmp_path: Path) -> None:
        config_path = tmp_path / "bad_model.yaml"
        config_path.write_text(yaml.dump({"model": {"channel_aggregation": "ccf"}}))

        with pytest.raises(ValueError, match=r"model\.channel_aggregation"):
            load_training_config(config_path)

    def test_to_dict_round_trips_through_the_schema(self) -> None:
        """Checkpoints store `to_dict()`. Resume rebuilds from it, so it has to validate."""
        config = load_training_config(Path("configs/train/base.yaml"))
        assert training_config_from_dict(config.to_dict()) == config

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
        assert config.model.output_dim == 512
        assert config.optimization.loss_type == "infonce"

    def test_yaml_extends(self, tmp_path: Path) -> None:
        base = {"model": {"output_dim": 256, "ccf_layers": 2}}
        child = {
            "extends": "base.yaml",
            "model": {"ccf_layers": 4},
        }
        (tmp_path / "base.yaml").write_text(yaml.dump(base))
        (tmp_path / "child.yaml").write_text(yaml.dump(child))

        config = load_training_config(tmp_path / "child.yaml")
        assert config.model.output_dim == 256  # from base
        assert config.model.ccf_layers == 4  # overridden by child

    def test_dotted_overrides_apply_after_extends(self, tmp_path: Path) -> None:
        base = {"dataset": {"batch_size": 32}}
        (tmp_path / "base.yaml").write_text(yaml.dump(base))
        child = {"extends": "base.yaml"}
        config_path = tmp_path / "child.yaml"
        config_path.write_text(yaml.dump(child))

        config = load_training_config(
            config_path,
            overrides=["dataset.batch_size=8", "runtime.max_train_steps=3"],
        )
        assert config.dataset.batch_size == 8
        assert config.runtime.max_train_steps == 3

    def test_dotted_override_malformed_no_equals(self, tmp_path: Path) -> None:
        config_path = tmp_path / "base.yaml"
        config_path.write_text(yaml.dump({}))

        with pytest.raises(ValueError, match="Malformed --set"):
            load_training_config(config_path, overrides=["dataset.batch_size"])

    def test_dotted_override_malformed_no_section(self, tmp_path: Path) -> None:
        config_path = tmp_path / "base.yaml"
        config_path.write_text(yaml.dump({}))

        with pytest.raises(ValueError, match="Malformed --set"):
            load_training_config(config_path, overrides=["batch_size=8"])


class TestTrainingConfigFromDict:
    """Tests for reconstructing configs from a plain dict (e.g. checkpoints)."""

    def test_rejects_unknown_keys(self) -> None:
        """`betas` is a CellCLIP key. Loading it here must name the offending path."""
        with pytest.raises(ValueError, match=r"optimization\.betas"):
            training_config_from_dict({"optimization": {"betas": [0.9, 0.999]}})
