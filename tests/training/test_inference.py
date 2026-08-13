"""Tests for `encode_wells`, the one well-encoding loop.

`morphoclip eval`, `morphoclip infer` and profile export each had their own
copy. They disagreed on two things that matter: whether uncached wells are
dropped, and whether the forward pass runs under autocast. Export ran in fp32
and writes its embeddings to disk, so a helpful `amp=True` default would have
changed every benchmark number downstream without failing anything.
"""

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from morphoclip.training.batch_correction import PlateOffsets
from morphoclip.training.inference import encode_wells
from tests.support.batches import make_batch, make_text_cache

DEVICE = torch.device("cpu")


class CountingEncoder(nn.Module):
    """Returns one row per well, holding that well's site count."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, features: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        del features
        return site_mask.sum(dim=1).float().unsqueeze(1).expand(-1, self.dim).clone()


class DoublingProjection(nn.Module):
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw * 2


class TestImageOnly:
    def test_rows_line_up_with_wells_across_batches(self) -> None:
        loader = [make_batch(["A01", "A02"]), make_batch(["A03"], plate="P2")]
        encoded = encode_wells(CountingEncoder(), loader, device=DEVICE)

        assert encoded.image.shape == (3, 4)
        assert encoded.wells == ["A01", "A02", "A03"]
        assert encoded.plates == ["P1", "P1", "P2"]
        assert [info.broad_sample for info in encoded.pert_infos] == [
            "BRD-A01",
            "BRD-A02",
            "BRD-A03",
        ]

    def test_no_text_projection_means_no_text(self) -> None:
        encoded = encode_wells(CountingEncoder(), [make_batch(["A01"])], device=DEVICE)
        assert encoded.text is None
        with pytest.raises(ValueError, match="without a text projection"):
            encoded.require_text()

    def test_an_empty_loader_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(ValueError, match="No wells to encode"):
            encode_wells(CountingEncoder(), [], device=DEVICE)

    def test_batches_emptied_by_filtering_are_skipped_not_concatenated(self) -> None:
        loader = [make_batch(["A01"]), make_batch(["A02"])]
        encoded = encode_wells(
            CountingEncoder(), loader, device=DEVICE, text_cache=make_text_cache(["BRD-A02"])
        )
        assert encoded.wells == ["A02"]
        assert encoded.skipped == 1


class TestTextCache:
    def test_no_cache_means_no_filtering(self) -> None:
        """Profile export passes no cache, and must keep every well including controls."""
        encoded = encode_wells(CountingEncoder(), [make_batch(["A01", "A02"])], device=DEVICE)
        assert encoded.wells == ["A01", "A02"]
        assert encoded.skipped == 0

    def test_uncached_wells_are_dropped_and_counted(self) -> None:
        encoded = encode_wells(
            CountingEncoder(),
            [make_batch(["A01", "A02", "A03"])],
            device=DEVICE,
            text_projection=DoublingProjection(),
            text_cache=make_text_cache(["BRD-A01", "BRD-A03"]),
        )
        assert encoded.wells == ["A01", "A03"]
        assert encoded.skipped == 1
        assert encoded.image.shape[0] == encoded.require_text().shape[0] == 2

    def test_text_rows_follow_the_surviving_wells(self) -> None:
        cache = make_text_cache(["BRD-A01", "BRD-A02"])
        encoded = encode_wells(
            CountingEncoder(),
            [make_batch(["A02"])],
            device=DEVICE,
            text_projection=DoublingProjection(),
            text_cache=cache,
        )
        expected = cache["embeddings"][cache["id_to_idx"]["BRD-A02"]] * 2
        assert torch.equal(encoded.require_text()[0], expected)


class TestPlateOffsets:
    """The one hook that carries CWA into eval diagnostics, infer and export."""

    def test_offsets_are_subtracted_and_the_result_renormalized(self) -> None:
        offsets = PlateOffsets({"P1": torch.tensor([2.0, 0.0, 0.0, 0.0])})
        encoded = encode_wells(
            CountingEncoder(), [make_batch(["A01"])], device=DEVICE, plate_offsets=offsets
        )
        # CountingEncoder emits (2, 2, 2, 2) for a two-site well.
        expected = torch.nn.functional.normalize(torch.tensor([[0.0, 2.0, 2.0, 2.0]]), dim=-1)
        torch.testing.assert_close(encoded.image, expected)

    def test_no_offsets_leaves_the_encoder_output_untouched(self) -> None:
        encoded = encode_wells(CountingEncoder(), [make_batch(["A01"])], device=DEVICE)
        torch.testing.assert_close(encoded.image, torch.full((1, 4), 2.0))


def test_the_amp_flag_reaches_autocast_and_defaults_to_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU always takes the nullcontext branch, so intercept the call instead."""
    requested: list[bool] = []

    def record(_device: torch.device, enabled: bool):
        requested.append(enabled)
        return nullcontext()

    monkeypatch.setattr("morphoclip.training.inference.autocast_context", record)
    encode_wells(CountingEncoder(), [make_batch(["A01"])], device=DEVICE)
    encode_wells(CountingEncoder(), [make_batch(["A01"])], device=DEVICE, amp=True)
    assert requested == [False, True]
