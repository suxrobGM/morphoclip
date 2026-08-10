"""Tests for the cpjump dataset config.

Seven call sites used to read configs/dataset.yml with their own defaults, so
`local.features` fell back to "data/features" in one place and raised a KeyError
in another. These pin the one set of defaults and the S3 URI assembly.
"""

from pathlib import Path

import pytest
import yaml

from morphoclip.data.config import CPJumpConfig, load_dataset_config

PROJECT_CONFIG = Path("configs/dataset.yml")


def write(path: Path, cpjump: dict) -> Path:
    path.write_text(yaml.dump({"cpjump": cpjump}), encoding="utf-8")
    return path


class TestShippedConfig:
    def test_the_repo_config_loads(self) -> None:
        config = load_dataset_config(PROJECT_CONFIG)
        assert config.batch == "2020_11_04_CPJUMP1"
        assert len(config.plates) == 51
        assert config.local.features == Path("data/features")

    def test_the_repo_config_sets_no_key_the_schema_rejects(self) -> None:
        """`extra="forbid"` means a stale key in dataset.yml fails loudly."""
        raw = yaml.safe_load(PROJECT_CONFIG.read_text(encoding="utf-8"))["cpjump"]
        assert set(raw) <= set(CPJumpConfig.model_fields)


class TestDefaults:
    def test_only_batch_is_required(self, tmp_path: Path) -> None:
        """A config for local-only work needs no S3 coordinates."""
        config = load_dataset_config(write(tmp_path / "d.yml", {"batch": "B"}))
        assert config.local.metadata == Path("data/metadata")
        assert config.extraction.model.startswith("facebook/dinov3")
        assert config.fetch.backend == "auto"

    def test_a_missing_batch_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="batch"):
            load_dataset_config(write(tmp_path / "d.yml", {}))

    def test_a_missing_cpjump_section_names_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "d.yml"
        path.write_text(yaml.dump({"other": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="no 'cpjump' section"):
            load_dataset_config(path)

    def test_an_unknown_key_names_its_own_path(self, tmp_path: Path) -> None:
        path = write(tmp_path / "d.yml", {"batch": "B", "local": {"typo": "x"}})
        with pytest.raises(ValueError, match=r"local\.typo"):
            load_dataset_config(path)

    def test_a_zero_extraction_batch_size_is_rejected(self, tmp_path: Path) -> None:
        path = write(tmp_path / "d.yml", {"batch": "B", "extraction": {"batch_size": 0}})
        with pytest.raises(ValueError, match="batch_size"):
            load_dataset_config(path)

    def test_paths_arrive_as_paths_not_strings(self, tmp_path: Path) -> None:
        path = write(tmp_path / "d.yml", {"batch": "B", "local": {"features": "/tmp/f"}})
        assert isinstance(load_dataset_config(path).local.features, Path)


class TestS3Uri:
    def test_the_batch_placeholder_is_expanded(self) -> None:
        config = CPJumpConfig(batch="2020_11_04_CPJUMP1")
        assert config.s3_uri(config.images) == (
            "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images"
        )

    def test_a_template_without_a_placeholder_is_left_alone(self) -> None:
        config = CPJumpConfig(batch="B", endpoint="s3://bucket")
        assert config.s3_uri("workspace/external") == "s3://bucket/workspace/external"

    def test_no_double_slash_when_the_endpoint_has_a_trailing_one(self) -> None:
        config = CPJumpConfig(batch="B", endpoint="s3://bucket/")
        assert config.s3_uri("/prefix/") == "s3://bucket/prefix"
