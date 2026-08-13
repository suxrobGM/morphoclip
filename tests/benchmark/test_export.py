"""Tests for the MorphoCLIP profile-export path.

Export is the only consumer of a checkpoint whose output is written to disk and
then scored, so a CWA correction silently dropped here would move every
benchmark number without failing anything. ``morphoclip.benchmark`` is first
party and not frozen; the standalone ``benchmark`` package stays untested.
"""

import dataclasses
from pathlib import Path

import pytest
import torch
from torch import nn

from morphoclip.benchmark.export import _encode_plate_wells, load_export_models
from morphoclip.data.metadata import MetadataIndex
from morphoclip.training.batch_correction import PlateOffsets
from morphoclip.training.config import training_config_from_dict
from morphoclip.training.engine import save_checkpoint
from morphoclip.training.inference import build_models
from morphoclip.training.optim import build_optimizer, build_warmup_cosine_scheduler, split_params
from tests.support.features import make_feature_root, write_dataset_yml

pytestmark = pytest.mark.slow

PLATE = "BR00116991"
OUTPUT_DIM = 8


@pytest.fixture
def checkpoint(tmp_path: Path, metadata_dir: Path) -> Path:
    """A checkpoint over two fake wells, carrying a known plate offset."""
    metadata = MetadataIndex.from_directory(metadata_dir, batch="2020_11_04_CPJUMP1")
    wells = sorted(metadata.wells_for_plate(PLATE))[:2]
    feature_root = make_feature_root(tmp_path / "features", {PLATE: wells})
    dataset_yml = write_dataset_yml(tmp_path / "dataset.yml", metadata_dir=metadata_dir)

    config = training_config_from_dict(
        {
            "dataset": {
                "dataset_config_path": str(dataset_yml),
                "feature_root": str(feature_root),
                "eval_batch_size": 2,
                "exclude_controls": False,
            },
            "model": {"output_dim": OUTPUT_DIM, "ccf_layers": 1, "ccf_heads": 2},
            "optimization": {"use_cwa": True},
            "runtime": {"device": "cpu", "amp": False},
        }
    )
    image_encoder, text_projection = build_models(config, torch.device("cpu"))
    logit_scale = nn.Parameter(torch.tensor(2.6593))
    optimizer = build_optimizer(
        split_params(image_encoder, text_projection, weight_decay=0.0), config
    )

    # Written through save_checkpoint, not hand-rolled: the payload keys are the
    # contract export replays, and a rename should fail here.
    path = tmp_path / "best.pt"
    save_checkpoint(
        path,
        image_encoder=image_encoder,
        text_projection=text_projection,
        logit_scale=logit_scale,
        optimizer=optimizer,
        scheduler=build_warmup_cosine_scheduler(optimizer, total_steps=1, warmup_steps=1),
        epoch=1,
        global_step=1,
        best_eval_loss=0.0,
        config=config,
        plate_offsets=PlateOffsets({PLATE: torch.linspace(-1.0, 1.0, OUTPUT_DIM)}),
    )
    return path


def test_export_applies_the_checkpoints_plate_offsets(checkpoint: Path) -> None:
    models = load_export_models(checkpoint, "cpu")
    assert models.plate_offsets is not None

    corrected, wells = _encode_plate_wells(models, PLATE, batch_size=None)
    raw, raw_wells = _encode_plate_wells(
        dataclasses.replace(models, plate_offsets=None), PLATE, batch_size=None
    )

    assert wells == raw_wells
    expected = models.plate_offsets.apply(torch.from_numpy(raw), [PLATE] * len(wells))
    torch.testing.assert_close(torch.from_numpy(corrected), expected)
