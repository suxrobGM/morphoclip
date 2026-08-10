"""Tests for the cpjump dataset config.

Seven call sites used to read configs/dataset.yml with their own defaults, so
`local.features` fell back to "data/features" in one place and raised a KeyError
in another. These pin the one set of defaults and the S3 URI assembly.
"""

from pathlib import Path

import pytest
import yaml

from morphoclip.data.config import CPJumpConfig, load_dataset_config
from tests.conftest import PROJECT_ROOT

PROJECT_CONFIG = PROJECT_ROOT / "configs" / "dataset.yml"


def write(path: Path, doc: dict) -> Path:
    path.write_text(yaml.dump(doc), encoding="utf-8")
    return path


def test_the_repo_config_loads() -> None:
    config = load_dataset_config(PROJECT_CONFIG)
    assert config.batch == "2020_11_04_CPJUMP1"
    assert len(config.plates) == 51
    assert config.local.features == Path("data/features")


def test_only_batch_is_required(tmp_path: Path) -> None:
    """A config for local-only work needs no S3 coordinates."""
    config = load_dataset_config(write(tmp_path / "d.yml", {"cpjump": {"batch": "B"}}))
    assert config.local.metadata == Path("data/metadata")
    assert config.extraction.model.startswith("facebook/dinov3")
    assert config.fetch.backend == "auto"


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"other": {}}, "no 'cpjump' section"),
        ({"cpjump": {}}, "batch"),
        ({"cpjump": {"batch": "B", "local": {"typo": "x"}}}, r"local\.typo"),
        ({"cpjump": {"batch": "B", "extraction": {"batch_size": 0}}}, "batch_size"),
    ],
    ids=["no-section", "no-batch", "unknown-key", "zero-batch-size"],
)
def test_a_broken_config_is_rejected_with_a_message_naming_the_problem(
    tmp_path: Path, doc: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        load_dataset_config(write(tmp_path / "d.yml", doc))


@pytest.mark.parametrize(
    ("endpoint", "template", "expected"),
    [
        (
            "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4",
            "images/{batch}/images",
            "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/B/images",
        ),
        ("s3://bucket", "workspace/external", "s3://bucket/workspace/external"),
        ("s3://bucket/", "/prefix/", "s3://bucket/prefix"),
    ],
    ids=["batch-placeholder", "no-placeholder", "trailing-slash"],
)
def test_an_s3_uri_expands_the_batch_and_joins_on_a_single_slash(
    endpoint: str, template: str, expected: str
) -> None:
    assert CPJumpConfig(batch="B", endpoint=endpoint).s3_uri(template) == expected
