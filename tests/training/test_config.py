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

    def test_default_creation(self) -> None:
        config = MorphoCLIPTrainingConfig()
        assert config.model.output_dim == 512
        assert config.model.aggregator == "ccf-mean"
        assert config.optimization.loss_type == "infonce"
        assert config.optimization.lr == 3.0e-4
        assert config.runtime.device == "auto"
        assert config.dataset.batch_size == 32

    def test_training_signal_fields_default_off(self) -> None:
        """New features must default to current behavior."""
        config = MorphoCLIPTrainingConfig()
        assert config.dataset.batch_sampler == "random"
        assert config.dataset.replicates_per_group == 2
        assert config.optimization.target_weight == 0.0
        assert config.optimization.lambda_img == 0.0
        assert config.optimization.img_img_temperature is None
        assert config.runtime.early_stop_patience is None

    def test_training_signal_fields_settable(self) -> None:
        config = training_config_from_dict(
            {
                "dataset": {"batch_sampler": "perturbation", "replicates_per_group": 4},
                "optimization": {
                    "target_weight": 0.6,
                    "lambda_img": 0.3,
                    "img_img_temperature": 0.1,
                },
                "runtime": {"early_stop_patience": 5},
            }
        )
        assert config.dataset.batch_sampler == "perturbation"
        assert config.dataset.replicates_per_group == 4
        assert config.optimization.target_weight == 0.6
        assert config.optimization.lambda_img == 0.3
        assert config.optimization.img_img_temperature == 0.1
        assert config.runtime.early_stop_patience == 5

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
        assert config.optimization.lambda_img == 0.0
        assert config.optimization.img_img_temperature is None
        assert config.runtime.early_stop_patience is None

    def test_mean_pool_yaml_uses_new_aggregator_field(self) -> None:
        mean_pool_yaml = Path("configs/train/mean_pool.yaml")
        if not mean_pool_yaml.exists():
            pytest.skip("configs/train/mean_pool.yaml not available")
        config = load_training_config(mean_pool_yaml)
        assert config.model.aggregator == "meanpool-mean"

    def test_legacy_channel_aggregation_rejected_in_strict_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "legacy.yaml"
        config_path.write_text(yaml.dump({"model": {"channel_aggregation": "ccf"}}))

        with pytest.raises(ValueError, match="Unknown config key"):
            load_training_config(config_path)

    def test_to_dict_roundtrip(self) -> None:
        config = MorphoCLIPTrainingConfig()
        d = config.to_dict()
        assert d["model"]["output_dim"] == 512
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

    def test_yaml_load_unknown_key_raises(self, tmp_path: Path) -> None:
        config_data = {"dataset": {"not_a_real_field": 1}}
        config_path = tmp_path / "bad_config.yaml"
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="Unknown config key"):
            load_training_config(config_path)

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

    def test_strict_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="Unknown config key"):
            training_config_from_dict(
                {"optimization": {"betas": [0.9, 0.999]}},
                strict=True,
            )

    def test_non_strict_skips_unknown_keys(self) -> None:
        # Mirrors an older checkpoint's embedded config dict, which may carry
        # fields dropped from the current schema (betas, eval_every_epochs,
        # embed_dim, tensorboard, distributed.backend, ...).
        old_dict = {
            "dataset": {"batch_size": 16, "max_train_wells": 100},
            "model": {"embed_dim": 1024, "output_dim": 512},
            "optimization": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "grad_clip_norm": 1.0,
                "logit_scale_max": 4.6052,
            },
            "runtime": {"seed": 7, "eval_every_epochs": 1, "save_every_epochs": 1},
            "tensorboard": {"histogram_every_epochs": 5},
            "distributed": {"enabled": False, "backend": "nccl"},
        }
        config = training_config_from_dict(old_dict, strict=False)
        assert config.dataset.batch_size == 16
        assert config.model.output_dim == 512
        assert config.optimization.lr == 1e-4
        assert config.runtime.seed == 7
        assert config.distributed.enabled is False

    @pytest.mark.parametrize(
        ("legacy", "expected"),
        [("ccf", "ccf-mean"), ("mean_pool", "meanpool-mean")],
    )
    def test_legacy_channel_aggregation_translated(self, legacy: str, expected: str) -> None:
        """Old checkpoints embed model.channel_aggregation; map it explicitly."""
        config = training_config_from_dict(
            {"model": {"channel_aggregation": legacy, "ccf_layers": 1}},
            strict=False,
        )
        assert config.model.aggregator == expected
        assert config.model.ccf_layers == 1

    def test_explicit_aggregator_wins_over_legacy_key(self) -> None:
        config = training_config_from_dict(
            {"model": {"channel_aggregation": "ccf", "aggregator": "wellformer"}},
            strict=False,
        )
        assert config.model.aggregator == "wellformer"

    def test_unrecognized_legacy_channel_aggregation_falls_back_to_default(self) -> None:
        config = training_config_from_dict(
            {"model": {"channel_aggregation": "mystery"}},
            strict=False,
        )
        assert config.model.aggregator == "ccf-mean"
