"""End-to-end characterization of the training entry point.

trainer.py, loop.py, engine.py, tb_logger.py and train_data.py are 1,229 lines
with no test between them, and they are the code path the project exists for.
This runs the real thing on CPU against fake features: no network, no GPU, a
couple of seconds.

It is deliberately shallow on numerics and strict on wiring. What it pins is
that optimizer, scaler, scheduler, checkpointing, resume and the metrics file
are still connected to each other after a refactor.
"""

import math
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import torch

from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import is_control_or_empty
from morphoclip.training.config import MorphoCLIPTrainingConfig, training_config_from_dict
from morphoclip.training.trainer import train_morphoclip
from tests.support.features import make_feature_root, write_dataset_yml, write_text_cache

pytestmark = pytest.mark.slow

PLATES = ("BR00116991", "BR00116992")
WELLS_PER_PLATE = 8


def _usable_wells(metadata: MetadataIndex, plate: str, count: int) -> list[str]:
    """Non-control wells, which is what the dataset keeps by default."""
    wells: list[str] = []
    for well in sorted(metadata.wells_for_plate(plate)):
        if not is_control_or_empty(metadata.lookup(plate, well)):
            wells.append(well)
        if len(wells) == count:
            break
    return wells


@pytest.fixture
def training_config(tmp_path: Path, metadata_dir: Path) -> Iterator[MorphoCLIPTrainingConfig]:
    metadata = MetadataIndex.from_directory(metadata_dir, batch="2020_11_04_CPJUMP1")
    plates = {plate: _usable_wells(metadata, plate, WELLS_PER_PLATE) for plate in PLATES}

    feature_root = make_feature_root(tmp_path / "features", plates, sites=2)
    text_cache = write_text_cache(tmp_path / "text.pt", metadata, plates)
    dataset_yml = write_dataset_yml(tmp_path / "dataset.yml", metadata_dir=metadata_dir)

    yield training_config_from_dict(
        {
            "dataset": {
                "dataset_config_path": str(dataset_yml),
                "feature_root": str(feature_root),
                "text_cache_path": str(text_cache),
                "batch_size": 4,
                "eval_batch_size": 4,
                "val_fraction": 0.25,
                "preload": True,
            },
            "model": {"output_dim": 8, "ccf_layers": 1, "ccf_heads": 2},
            "optimization": {"epochs": 1, "warmup_steps": 1},
            "runtime": {
                "device": "cpu",
                "amp": False,
                "seed": 0,
                "output_root": str(tmp_path / "runs"),
            },
        }
    )


def test_training_writes_a_checkpoint_and_finite_metrics(
    training_config: MorphoCLIPTrainingConfig, tmp_path: Path
) -> None:
    run_dir = tmp_path / "runs" / "smoke"
    summary = train_morphoclip(training_config, run_dir=run_dir)

    assert set(summary) >= {
        "run_dir",
        "train_wells",
        "val_wells",
        "best_checkpoint",
        "last_checkpoint",
        "metrics_path",
    }
    assert summary["train_wells"] > 0
    assert summary["val_wells"] > 0
    assert Path(summary["best_checkpoint"]).exists()
    assert Path(summary["last_checkpoint"]).exists()

    history = pd.read_csv(summary["metrics_path"])
    assert not history.empty, "trainer wrote no epoch history"
    assert math.isfinite(history["train_loss"].iloc[0])
    assert math.isfinite(history["eval_loss"].iloc[0])


def test_checkpoint_carries_everything_resume_needs(
    training_config: MorphoCLIPTrainingConfig, tmp_path: Path
) -> None:
    """The checkpoint payload is the contract between a run and its resume."""
    summary = train_morphoclip(training_config, run_dir=tmp_path / "runs" / "keys")
    checkpoint = torch.load(summary["last_checkpoint"], map_location="cpu", weights_only=False)

    assert set(checkpoint) >= {
        "image_encoder",
        "text_projection",
        "logit_scale",
        "optimizer",
        "scheduler",
        "epoch",
        "config",
    }
    # The embedded config must survive the round trip that resume depends on.
    assert training_config_from_dict(checkpoint["config"]).model.output_dim == 8


def test_resume_continues_instead_of_restarting(
    training_config: MorphoCLIPTrainingConfig, tmp_path: Path
) -> None:
    first = train_morphoclip(training_config, run_dir=tmp_path / "runs" / "first")
    epoch_after_first = torch.load(
        first["last_checkpoint"], map_location="cpu", weights_only=False
    )["epoch"]

    training_config.optimization.epochs = 2
    second = train_morphoclip(
        training_config,
        run_dir=tmp_path / "runs" / "second",
        resume_from=Path(first["last_checkpoint"]),
    )
    epoch_after_resume = torch.load(
        second["last_checkpoint"], map_location="cpu", weights_only=False
    )["epoch"]

    assert epoch_after_resume > epoch_after_first
