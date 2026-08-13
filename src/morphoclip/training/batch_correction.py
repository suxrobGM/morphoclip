"""Cross-Well Alignment (CWA): condition-relative plate offsets.

CWA used to subtract the mean of whichever wells of a plate happened to land in
the current batch. On CPJUMP1 that is wrong twice over. A plate is very nearly a
single condition (one cell line, one modality, one timepoint), so the full plate
mean carries exactly the biology the text prompts describe, and subtracting it
deletes the training signal. The estimate was also drawn from a handful of wells
and resampled every step, so most of what it removed was perturbation biology
plus sampling noise.

What replaces it is a precomputed per-plate offset,
``offset(plate) = plate mean - condition mean``, where the condition mean is the
mean of the per-plate means over the replicate plates sharing that plate's
condition. Offsets within a condition sum to zero by construction, so the
correction removes replicate-to-replicate drift and leaves the condition
untouched. They are recomputed once per epoch and treated as constants, never as
a differentiable path.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from morphoclip.data.perturbation import extract_plate_barcode
from morphoclip.training.distributed import DistributedState, broadcast_tensor
from morphoclip.training.optim import unwrap

logger = logging.getLogger(__name__)


def _require_row_alignment(embeddings: torch.Tensor, plates: list[str]) -> None:
    """Raise unless there is exactly one plate name per embedding row."""
    if len(plates) != embeddings.shape[0]:
        raise ValueError(
            f"plates length ({len(plates)}) does not match embeddings rows ({embeddings.shape[0]})"
        )


class PlateOffsets:
    """Per-plate correction vectors, keyed by plate barcode.

    Args:
        offsets: Plate barcode to a ``(D,)`` offset vector.
    """

    def __init__(self, offsets: dict[str, torch.Tensor]) -> None:
        self.offsets = offsets
        self._warned: set[str] = set()

    def _lookup(self, plate: str) -> torch.Tensor | None:
        """Offset for a plate name, or ``None`` after warning once about it.

        ``batch["plates"]`` holds feature-directory names such as
        ``BR00116991__2020-11-05T19_51_35-Measurement1`` while the table is keyed
        by barcode, so every lookup normalizes first.
        """
        barcode = extract_plate_barcode(plate)
        offset = self.offsets.get(barcode)
        if offset is None and barcode not in self._warned:
            self._warned.add(barcode)
            logger.warning("No plate offset for %r; CWA leaves its embeddings uncorrected", barcode)
        return offset

    def apply(self, embeddings: torch.Tensor, plates: list[str]) -> torch.Tensor:
        """Subtract each row's plate offset and re-normalize to the unit sphere.

        Args:
            embeddings: ``(B, D)`` image embeddings.
            plates: Plate name per row, either a barcode or a feature-directory
                name (length B).

        Returns:
            Corrected ``(B, D)`` embeddings, L2-normalized. The input is not
            mutated.

        Raises:
            ValueError: If ``plates`` is not one entry per embedding row.
        """
        _require_row_alignment(embeddings, plates)

        zero = torch.zeros(embeddings.shape[1], dtype=torch.float32)
        rows = [self._lookup(plate) for plate in plates]
        stacked = torch.stack([zero if row is None else row for row in rows]).to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        return F.normalize(embeddings - stacked, dim=-1)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Checkpoint payload: fp32 CPU tensors keyed by plate barcode."""
        return {
            plate: offset.detach().to("cpu", torch.float32)
            for plate, offset in self.offsets.items()
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> PlateOffsets:
        """Rebuild from a checkpoint payload written by :meth:`state_dict`."""
        return cls({plate: offset.to("cpu", torch.float32) for plate, offset in state.items()})


def compute_plate_offsets(
    embeddings: torch.Tensor,
    plates: list[str],
    plate_conditions: dict[str, str],
) -> PlateOffsets:
    """Compute ``plate mean - condition mean`` for every plate present.

    The condition mean weights each member plate equally rather than each well,
    so offsets within one condition sum to zero however unevenly the wells are
    distributed.

    Args:
        embeddings: ``(N, D)`` uncorrected well embeddings.
        plates: Plate name per row, barcode or feature-directory name.
        plate_conditions: Plate barcode to condition key, from
            :func:`morphoclip.splits.contexts.load_plate_conditions`.

    Returns:
        Offsets for every plate seen, zero for plates with no known condition.

    Raises:
        ValueError: If ``plates`` is not one entry per embedding row.
    """
    _require_row_alignment(embeddings, plates)

    features = embeddings.detach().to("cpu", torch.float32)
    rows_by_plate: dict[str, list[int]] = {}
    for i, plate in enumerate(plates):
        rows_by_plate.setdefault(extract_plate_barcode(plate), []).append(i)

    plate_means = {barcode: features[rows].mean(dim=0) for barcode, rows in rows_by_plate.items()}

    plates_by_condition: dict[str, list[str]] = {}
    unknown: list[str] = []
    for barcode in plate_means:
        condition = plate_conditions.get(barcode)
        if condition is None:
            unknown.append(barcode)
            continue
        plates_by_condition.setdefault(condition, []).append(barcode)
    if unknown:
        logger.warning(
            "%d plate(s) absent from the condition map get a zero CWA offset: %s",
            len(unknown),
            sorted(unknown)[:10],
        )

    offsets = {barcode: torch.zeros_like(plate_means[barcode]) for barcode in unknown}
    for members in plates_by_condition.values():
        condition_mean = torch.stack([plate_means[barcode] for barcode in members]).mean(dim=0)
        for barcode in members:
            offsets[barcode] = plate_means[barcode] - condition_mean
    return PlateOffsets(offsets)


def refresh_plate_offsets(
    image_encoder: nn.Module,
    loader,
    plate_conditions: dict[str, str],
    *,
    device: torch.device,
    amp: bool = False,
    dist_state: DistributedState | None = None,
) -> PlateOffsets:
    """Re-estimate plate offsets from a full pass over the training wells.

    Runs the encoder in eval mode under ``no_grad``, which is what keeps dropout
    out of the estimate, then restores the mode it found. Under DDP every rank
    runs the same unsharded pass and rank 0's table is broadcast, so all ranks
    hold bitwise identical offsets.

    Args:
        image_encoder: Image encoder, possibly DDP-wrapped.
        loader: Sequential DataLoader over the training wells.
        plate_conditions: Plate barcode to condition key.
        device: Device to run the pass on.
        amp: Run the pass under autocast, as the training forward does. This is
            a whole extra pass over the training set once per epoch, and fp32
            roughly doubles it. An offset is a mean over hundreds of wells, so
            autocast moves it by a fraction of a percent of its own magnitude.
        dist_state: Distributed state, when training under DDP.

    Returns:
        Freshly computed offsets, frozen for the coming epoch.
    """
    from morphoclip.training.inference import encode_wells

    module = unwrap(image_encoder)
    was_training = module.training
    module.eval()
    try:
        encoded = encode_wells(module, loader, device=device, amp=amp)
    finally:
        module.train(was_training)

    offsets = compute_plate_offsets(encoded.image, encoded.plates, plate_conditions)
    return _broadcast_offsets(offsets, dist_state)


def _broadcast_offsets(
    offsets: PlateOffsets,
    dist_state: DistributedState | None,
) -> PlateOffsets:
    """Replace every rank's table with rank 0's, so the ranks cannot drift apart."""
    if dist_state is None or dist_state.world_size < 2:
        return offsets

    barcodes = sorted(offsets.offsets)
    stacked = torch.stack([offsets.offsets[barcode] for barcode in barcodes]).to(dist_state.device)
    rows = broadcast_tensor(stacked).to("cpu", torch.float32)
    return PlateOffsets({barcode: rows[i] for i, barcode in enumerate(barcodes)})


def offsets_from_checkpoint(ckpt: dict, *, use_cwa: bool) -> PlateOffsets | None:
    """Read the plate offsets a checkpoint was saved with.

    Args:
        ckpt: Loaded checkpoint dict.
        use_cwa: Whether the checkpoint's run trained with CWA on. Only decides
            whether a missing table is worth warning about.

    Returns:
        The saved offsets, or ``None`` when CWA was off or the checkpoint
        predates offset persistence.
    """
    state = ckpt.get("plate_offsets")
    if state is None:
        if use_cwa:
            logger.warning(
                "Checkpoint has use_cwa=true but no 'plate_offsets'; "
                "evaluating uncorrected embeddings"
            )
        return None
    return PlateOffsets.from_state_dict(state)
