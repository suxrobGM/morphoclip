"""Tests for condition-relative CWA plate offsets.

The old in-batch version subtracted the full plate mean, which on CPJUMP1 is
very nearly the condition mean and so deleted the signal the prompts encode.
What these pin is that the replacement subtracts only the part of a plate mean
that its replicate plates disagree about, and that the barcode normalization
between feature-directory names and the offset table stays wired.
"""

import logging

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from morphoclip.training.batch_correction import (
    PlateOffsets,
    compute_plate_offsets,
    offsets_from_checkpoint,
    refresh_plate_offsets,
)
from tests.support.batches import make_batch
from tests.support.constants import HIDDEN_DIM

DEVICE = torch.device("cpu")
CONDITIONS = {"P1": "A549|Compound|24", "P2": "A549|Compound|24"}


def _embeddings(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


class TestComputePlateOffsets:
    def test_offset_is_the_plate_mean_minus_the_condition_mean(self) -> None:
        embeddings = _embeddings([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0], [7.0, 0.0]])
        offsets = compute_plate_offsets(embeddings, ["P1", "P1", "P2", "P2"], CONDITIONS)

        # Plate means 2 and 6, condition mean 4.
        torch.testing.assert_close(offsets.offsets["P1"], torch.tensor([-2.0, 0.0]))
        torch.testing.assert_close(offsets.offsets["P2"], torch.tensor([2.0, 0.0]))

    def test_offsets_in_one_condition_sum_to_zero_despite_unequal_well_counts(self) -> None:
        """Each plate weighs the same, so a plate with more wells cannot drag the mean."""
        embeddings = _embeddings([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [9.0, 0.0]])
        offsets = compute_plate_offsets(embeddings, ["P1", "P1", "P1", "P2"], CONDITIONS)

        total = offsets.offsets["P1"] + offsets.offsets["P2"]
        torch.testing.assert_close(total, torch.zeros(2))
        torch.testing.assert_close(offsets.offsets["P1"], torch.tensor([-4.0, 0.0]))

    def test_a_plate_with_no_known_condition_gets_a_zero_offset(self) -> None:
        embeddings = _embeddings([[1.0, 0.0], [3.0, 0.0], [5.0, 5.0]])
        offsets = compute_plate_offsets(embeddings, ["P1", "P2", "UNMAPPED"], CONDITIONS)

        torch.testing.assert_close(offsets.offsets["UNMAPPED"], torch.zeros(2))

    def test_a_row_count_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match embeddings rows"):
            compute_plate_offsets(_embeddings([[1.0, 0.0]]), ["P1", "P2"], CONDITIONS)


class TestApply:
    def test_the_corrected_output_lands_back_on_the_unit_sphere(self) -> None:
        offsets = PlateOffsets({"P1": torch.full((HIDDEN_DIM,), 0.5)})
        embeddings = F.normalize(torch.randn(4, HIDDEN_DIM), dim=-1)

        out = offsets.apply(embeddings, ["P1"] * 4)

        torch.testing.assert_close(out.norm(dim=-1), torch.ones(4), atol=1e-5, rtol=0)

    def test_the_input_is_not_mutated(self) -> None:
        offsets = PlateOffsets({"P1": torch.full((HIDDEN_DIM,), 0.5)})
        embeddings = F.normalize(torch.randn(4, HIDDEN_DIM), dim=-1)
        before = embeddings.clone()

        offsets.apply(embeddings, ["P1"] * 4)

        torch.testing.assert_close(embeddings, before)

    def test_feature_directory_names_resolve_to_barcode_keyed_offsets(self) -> None:
        """``batch["plates"]`` carries directory names; the table is keyed by barcode."""
        offsets = PlateOffsets({"BR00116991": _embeddings([[1.0, 0.0]])[0]})
        embeddings = _embeddings([[1.0, 1.0], [1.0, 1.0]])

        out = offsets.apply(
            embeddings,
            ["BR00116991__2020-11-05T19_51_35-Measurement1", "BR00116991"],
        )

        # An unresolved directory name would fall through to a zero offset and
        # leave both rows at (0.707, 0.707) instead.
        torch.testing.assert_close(out, _embeddings([[0.0, 1.0], [0.0, 1.0]]))

    def test_an_unknown_plate_is_left_alone_and_warned_about_exactly_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        offsets = PlateOffsets({"P1": torch.full((HIDDEN_DIM,), 0.5)})
        embeddings = F.normalize(torch.randn(3, HIDDEN_DIM), dim=-1)

        with caplog.at_level(logging.WARNING, logger="morphoclip.training.batch_correction"):
            out = offsets.apply(embeddings, ["MISSING", "MISSING", "MISSING"])

        assert sum("MISSING" in record.message for record in caplog.records) == 1
        torch.testing.assert_close(out, embeddings)

    def test_a_row_count_mismatch_is_rejected(self) -> None:
        offsets = PlateOffsets({"P1": torch.zeros(HIDDEN_DIM)})
        with pytest.raises(ValueError, match="does not match embeddings rows"):
            offsets.apply(torch.zeros(2, HIDDEN_DIM), ["P1"])


def test_a_state_dict_round_trip_corrects_identically() -> None:
    """The checkpoint payload is what eval, infer and export replay."""
    offsets = PlateOffsets({"P1": torch.randn(HIDDEN_DIM), "P2": torch.randn(HIDDEN_DIM)})
    embeddings = F.normalize(torch.randn(4, HIDDEN_DIM), dim=-1)
    plates = ["P1", "P2", "P1", "P2"]

    restored = PlateOffsets.from_state_dict(offsets.state_dict())

    torch.testing.assert_close(
        restored.apply(embeddings, plates), offsets.apply(embeddings, plates)
    )


class ModeRecordingEncoder(nn.Module):
    """Reports its own training flag, so the caller can see which mode ran."""

    def __init__(self) -> None:
        super().__init__()
        self.modes: list[bool] = []

    def forward(self, features: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        del site_mask
        self.modes.append(self.training)
        return torch.ones(features.shape[0], 2)


def test_the_refresh_pass_runs_in_eval_mode_and_restores_the_mode_it_found() -> None:
    """Dropout during the offset pass would poison every offset with noise."""
    encoder = ModeRecordingEncoder()
    encoder.train()
    loader = [
        make_batch(["A00", "A01"], plate="P1", sites=1),
        make_batch(["A02", "A03"], plate="P2", sites=1),
    ]

    refresh_plate_offsets(encoder, loader, CONDITIONS, device=DEVICE)

    assert encoder.modes == [False, False]
    assert encoder.training is True


class TestOffsetsFromCheckpoint:
    def test_a_saved_table_round_trips(self) -> None:
        saved = PlateOffsets({"P1": torch.zeros(HIDDEN_DIM)})
        loaded = offsets_from_checkpoint({"plate_offsets": saved.state_dict()}, use_cwa=True)
        assert loaded is not None
        assert set(loaded.offsets) == {"P1"}

    def test_a_cwa_checkpoint_without_offsets_warns_and_evaluates_uncorrected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="morphoclip.training.batch_correction"):
            assert offsets_from_checkpoint({}, use_cwa=True) is None
        assert any("plate_offsets" in record.message for record in caplog.records)

    def test_a_non_cwa_checkpoint_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="morphoclip.training.batch_correction"):
            assert offsets_from_checkpoint({"plate_offsets": None}, use_cwa=False) is None
        assert caplog.records == []
