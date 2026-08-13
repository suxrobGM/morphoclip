"""Tests for the evaluation loop's CWA hook.

`evaluate_epoch` used to take a ``use_cwa`` flag and rebuild the correction from
whatever wells were in the batch. It now applies the checkpoint's offsets, and
the one thing that must not rot is that they actually reach the embeddings the
eval loss is computed from.
"""

import torch
import torch.nn.functional as F
from torch import nn

from morphoclip.training.batch_correction import PlateOffsets
from morphoclip.training.evaluate import evaluate_epoch
from tests.support.batches import make_batch, make_text_cache

DEVICE = torch.device("cpu")
DIM = 4
WELLS = ["A01", "A02", "A03", "A04"]


class SpreadEncoder(nn.Module):
    """One distinct unit-norm embedding per well, so retrieval is not degenerate."""

    def forward(self, features: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        del site_mask
        return F.normalize(features[:, 0, 0, :DIM], dim=-1)


class IdentityProjection(nn.Module):
    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        return F.normalize(raw, dim=-1)


def _loader() -> list[dict]:
    generator = torch.Generator().manual_seed(0)
    return [
        make_batch(
            WELLS,
            plate=["P1", "P1", "P2", "P2"],
            sites=1,
            features=torch.randn(len(WELLS), 1, 1, DIM, generator=generator),
        )
    ]


def _eval_loss(plate_offsets: PlateOffsets | None) -> float:
    return evaluate_epoch(
        SpreadEncoder(),
        IdentityProjection(),
        make_text_cache([f"BRD-{well}" for well in WELLS], dim=DIM),
        _loader(),
        device=DEVICE,
        logit_scale=nn.Parameter(torch.tensor(2.6593)),
        loss_type="infonce",
        amp=False,
        plate_offsets=plate_offsets,
    )["eval_loss"]


def test_offsets_change_the_eval_loss_they_are_applied_to() -> None:
    """A dropped `plate_offsets.apply` would score the uncorrected embeddings."""
    offsets = PlateOffsets(
        {"P1": torch.full((DIM,), 5.0), "P2": torch.full((DIM,), -5.0)},
    )

    assert _eval_loss(offsets) != _eval_loss(None)
