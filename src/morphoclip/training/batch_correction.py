"""Cross-Well Alignment (CWA) batch correction for image embeddings.

Removes plate-level technical artifacts by subtracting per-plate mean
embeddings and re-normalizing to the unit sphere.  Applied to image
embeddings only — text embeddings have no batch effect.
"""

import torch
import torch.nn.functional as F


def cross_well_alignment(
    embeddings: torch.Tensor,
    plate_ids: list[str],
) -> torch.Tensor:
    """Remove plate batch effects via per-plate mean subtraction.

    For each plate represented in the batch, computes the mean embedding
    of all wells from that plate, subtracts it, and L2-normalizes the
    corrected embeddings.

    Args:
        embeddings: ``(B, D)`` embeddings (typically L2-normalized).
        plate_ids: Plate barcode for each sample (length B).

    Returns:
        Batch-corrected embeddings ``(B, D)``, L2-normalized.
    """
    corrected = embeddings.clone()

    # Build plate -> indices mapping
    plate_to_indices: dict[str, list[int]] = {}
    for i, plate in enumerate(plate_ids):
        plate_to_indices.setdefault(plate, []).append(i)

    # Subtract per-plate mean
    for indices in plate_to_indices.values():
        idx = torch.tensor(indices, device=embeddings.device)
        plate_mean = embeddings[idx].mean(dim=0, keepdim=True)
        corrected[idx] = corrected[idx] - plate_mean

    # Re-normalize to unit sphere
    return F.normalize(corrected, dim=-1)
