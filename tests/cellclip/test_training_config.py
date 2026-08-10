"""Tests for CellCLIP training config loading and validation.

The validation rules here used to be a 46-line imperative block at the end of
the loader with no test of its own. Each rule now has a case, because CellCLIP
is frozen: a rule that quietly stops firing would change a baseline nobody is
watching.
"""

from pathlib import Path

import pytest
import yaml

from cellclip.training.config import CellCLIPTrainingConfig, load_training_config


def write_config(path: Path, data: object) -> Path:
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


def test_load_training_config_supports_extends(tmp_path: Path) -> None:
    write_config(
        tmp_path / "base.yaml",
        {
            "dataset": {"feature_root": "data/features_cellclip_base", "batch_size": 16},
            "optimization": {"lr": 5.0e-4, "warmup_steps": 100},
            "runtime": {"run_name": "base_run"},
        },
    )
    child = write_config(
        tmp_path / "child.yaml",
        {
            "extends": "base.yaml",
            "dataset": {"batch_size": 64},
            "optimization": {"warmup_steps": 1000},
        },
    )

    config = load_training_config(child)

    assert config.dataset.feature_root == "data/features_cellclip_base"
    assert config.dataset.batch_size == 64
    assert config.optimization.lr == pytest.approx(5.0e-4)
    assert config.optimization.warmup_steps == 1000
    assert config.runtime.run_name == "base_run"


class TestSubsetNormalization:
    @pytest.mark.parametrize(
        ("written", "expected"),
        [("val", "validate"), ("VALIDATE", "validate"), (" train ", "train"), ("test", "test")],
    )
    def test_subset_aliases_become_manifest_labels(
        self, tmp_path: Path, written: str, expected: str
    ) -> None:
        path = write_config(tmp_path / "c.yaml", {"dataset": {"subset": written}})
        assert load_training_config(path).dataset.subset == expected

    def test_an_unknown_subset_is_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path / "c.yaml", {"dataset": {"eval_subset": "holdout"}})
        with pytest.raises(ValueError, match="Unknown split subset"):
            load_training_config(path)


class TestModelChoices:
    def test_choice_fields_fold_case(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path / "c.yaml",
            {"model": {"variant": " ChemBERTa ", "chemberta_pooling": "CLS"}},
        )
        config = load_training_config(path)
        assert config.model.variant == "chemberta"
        assert config.model.chemberta_pooling == "cls"

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("variant", "sideways"),
            ("chemberta_pooling", "max"),
            ("chem_fusion_type", "gated"),
            ("chem_prompt_policy", "drop_smiles"),
        ],
    )
    def test_an_unknown_choice_names_the_field(
        self, tmp_path: Path, field: str, value: str
    ) -> None:
        path = write_config(tmp_path / "c.yaml", {"model": {field: value}})
        with pytest.raises(ValueError, match=field):
            load_training_config(path)

    def test_the_film_variant_overrides_an_explicit_fusion_type(self, tmp_path: Path) -> None:
        """`chemberta_film` names its own fusion, so a conflicting setting loses."""
        path = write_config(
            tmp_path / "c.yaml",
            {"model": {"variant": "chemberta_film", "chem_fusion_type": "concat_mlp"}},
        )
        assert load_training_config(path).model.chem_fusion_type == "film"

    def test_the_film_variant_does_not_launder_a_bogus_fusion_type(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path / "c.yaml",
            {"model": {"variant": "chemberta_film", "chem_fusion_type": "gated"}},
        )
        with pytest.raises(ValueError, match="chem_fusion_type"):
            load_training_config(path)

    def test_tuning_layers_of_a_frozen_encoder_is_rejected(self, tmp_path: Path) -> None:
        path = write_config(
            tmp_path / "c.yaml",
            {"model": {"freeze_chemberta": True, "chemberta_tune_layers": 2}},
        )
        with pytest.raises(ValueError, match="freeze_chemberta"):
            load_training_config(path)

    def test_negative_tune_layers_are_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path / "c.yaml", {"model": {"chemberta_tune_layers": -1}})
        with pytest.raises(ValueError, match="chemberta_tune_layers"):
            load_training_config(path)


class TestDatasetBounds:
    @pytest.mark.parametrize(
        "field", ["max_sites_per_well", "train_max_sites_per_well", "eval_max_sites_per_well"]
    )
    def test_a_site_cap_of_zero_is_rejected(self, tmp_path: Path, field: str) -> None:
        path = write_config(tmp_path / "c.yaml", {"dataset": {field: 0}})
        with pytest.raises(ValueError, match=field):
            load_training_config(path)

    @pytest.mark.parametrize(
        "field", ["max_sites_per_well", "train_max_sites_per_well", "eval_max_sites_per_well"]
    )
    def test_an_unset_site_cap_stays_none(self, tmp_path: Path, field: str) -> None:
        path = write_config(tmp_path / "c.yaml", {"dataset": {field: None}})
        assert getattr(load_training_config(path).dataset, field) is None

    @pytest.mark.parametrize("field", ["within_well_interp_sites", "same_pert_interp_sites"])
    def test_negative_interpolation_counts_are_rejected(self, tmp_path: Path, field: str) -> None:
        path = write_config(tmp_path / "c.yaml", {"dataset": {field: -1}})
        with pytest.raises(ValueError, match=field):
            load_training_config(path)

    def test_a_zero_interpolation_alpha_is_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path / "c.yaml", {"dataset": {"interp_alpha": 0}})
        with pytest.raises(ValueError, match="interp_alpha"):
            load_training_config(path)


class TestOptimization:
    def test_betas_arrive_as_a_tuple_for_adamw(self, tmp_path: Path) -> None:
        path = write_config(tmp_path / "c.yaml", {"optimization": {"betas": [0.8, 0.95]}})
        assert load_training_config(path).optimization.betas == (0.8, 0.95)

    def test_three_betas_are_rejected(self, tmp_path: Path) -> None:
        path = write_config(tmp_path / "c.yaml", {"optimization": {"betas": [0.9, 0.99, 0.999]}})
        with pytest.raises(ValueError, match="betas"):
            load_training_config(path)


def test_to_dict_round_trips_through_the_schema() -> None:
    """Checkpoints store `to_dict()`. Resume rebuilds from it, so it has to validate."""
    config = CellCLIPTrainingConfig()
    assert CellCLIPTrainingConfig.model_validate(config.to_dict()) == config
