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

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType
from morphoclip.training.inference import EncodedWells, encode_wells

DEVICE = torch.device("cpu")


class RecordingEncoder(nn.Module):
    """Returns one row per well, and remembers the dtype it was called in."""

    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.autocast_was_enabled: list[bool] = []

    def forward(self, features: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        self.autocast_was_enabled.append(torch.is_autocast_enabled("cpu"))
        counts = site_mask.sum(dim=1).float().unsqueeze(1)
        return counts.expand(-1, self.dim).clone()


class DoublingProjection(nn.Module):
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return raw * 2


def make_batch(wells: list[str], *, plate: str = "P1", sites: int = 2) -> dict:
    pert_infos = [
        PerturbationInfo(pert_type=PerturbationType.COMPOUND, broad_sample=f"BRD-{well}")
        for well in wells
    ]
    return {
        "features": torch.ones(len(wells), sites, 5, 8),
        "site_mask": torch.ones(len(wells), sites, dtype=torch.bool),
        "plates": [plate] * len(wells),
        "wells": list(wells),
        "pert_info": pert_infos,
    }


def make_text_cache(broad_samples: list[str], dim: int = 4) -> dict:
    return {
        "id_to_idx": {sample: i for i, sample in enumerate(broad_samples)},
        "embeddings": torch.arange(len(broad_samples) * dim, dtype=torch.float).reshape(-1, dim),
        "perturbation_ids": list(broad_samples),
    }


class TestImageOnly:
    def test_rows_line_up_with_wells_across_batches(self) -> None:
        loader = [make_batch(["A01", "A02"]), make_batch(["A03"], plate="P2")]
        encoded = encode_wells(RecordingEncoder(), loader, device=DEVICE)

        assert encoded.image.shape == (3, 4)
        assert encoded.wells == ["A01", "A02", "A03"]
        assert encoded.plates == ["P1", "P1", "P2"]
        assert [info.broad_sample for info in encoded.pert_infos] == [
            "BRD-A01",
            "BRD-A02",
            "BRD-A03",
        ]

    def test_no_text_projection_means_no_text(self) -> None:
        encoded = encode_wells(RecordingEncoder(), [make_batch(["A01"])], device=DEVICE)
        assert encoded.text is None
        with pytest.raises(ValueError, match="without a text projection"):
            encoded.require_text()

    def test_an_empty_loader_raises_rather_than_returning_nothing(self) -> None:
        with pytest.raises(ValueError, match="No wells to encode"):
            encode_wells(RecordingEncoder(), [], device=DEVICE)

    def test_batches_emptied_by_filtering_are_skipped_not_concatenated(self) -> None:
        loader = [make_batch(["A01"]), make_batch(["A02"])]
        encoded = encode_wells(
            RecordingEncoder(),
            loader,
            device=DEVICE,
            text_cache=make_text_cache(["BRD-A02"]),
        )
        assert encoded.wells == ["A02"]
        assert encoded.skipped == 1


class TestTextCache:
    def test_no_cache_means_no_filtering(self) -> None:
        """Profile export passes no cache, and must keep every well including controls."""
        encoded = encode_wells(RecordingEncoder(), [make_batch(["A01", "A02"])], device=DEVICE)
        assert encoded.wells == ["A01", "A02"]
        assert encoded.skipped == 0

    def test_uncached_wells_are_dropped_and_counted(self) -> None:
        loader = [make_batch(["A01", "A02", "A03"])]
        encoded = encode_wells(
            RecordingEncoder(),
            loader,
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
            RecordingEncoder(),
            [make_batch(["A02"])],
            device=DEVICE,
            text_projection=DoublingProjection(),
            text_cache=cache,
        )
        expected = cache["embeddings"][cache["id_to_idx"]["BRD-A02"]] * 2
        assert torch.equal(encoded.require_text()[0], expected)


class TestAutocast:
    def test_amp_is_off_unless_asked_for(self) -> None:
        """The default protects exported profiles from silently becoming fp16."""
        encoder = RecordingEncoder()
        encode_wells(encoder, [make_batch(["A01"])], device=DEVICE)
        assert encoder.autocast_was_enabled == [False]

    def test_the_amp_flag_reaches_autocast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CPU always takes the nullcontext branch, so intercept the call instead."""
        requested: list[bool] = []

        def record(_device: torch.device, enabled: bool):
            requested.append(enabled)
            return nullcontext()

        monkeypatch.setattr("morphoclip.training.inference.autocast_context", record)
        encode_wells(RecordingEncoder(), [make_batch(["A01"])], device=DEVICE)
        encode_wells(RecordingEncoder(), [make_batch(["A01"])], device=DEVICE, amp=True)
        assert requested == [False, True]


def test_encoded_wells_defaults_are_not_shared_between_instances() -> None:
    first, second = EncodedWells(image=torch.empty(0)), EncodedWells(image=torch.empty(0))
    first.wells.append("A01")
    assert second.wells == []
