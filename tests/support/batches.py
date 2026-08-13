"""Builders for the batch and text-cache dicts the training loops consume.

``make_batch`` mirrors what :func:`morphoclip.data.dataset.collate_fn` emits, so
a renamed key breaks every test that feeds a loop instead of only the ones that
happen to touch that key.
"""

import torch

from morphoclip.data.perturbation import PerturbationInfo, PerturbationType


def make_batch(
    wells: list[str],
    *,
    plate: str | list[str] = "P1",
    sites: int = 2,
    features: torch.Tensor | None = None,
) -> dict:
    """Build one collated batch over *wells*.

    Args:
        wells: Well positions, one per row.
        plate: One plate name for every row, or a name per row.
        sites: Sites per well.
        features: Override the ``(wells, sites, 5, 8)`` ones tensor, for
            encoders that read the feature values rather than their shape.

    Returns:
        A dict with the keys ``collate_fn`` produces.
    """
    plates = [plate] * len(wells) if isinstance(plate, str) else list(plate)
    if len(plates) != len(wells):
        raise ValueError(f"plate list length ({len(plates)}) does not match wells ({len(wells)})")
    return {
        "features": torch.ones(len(wells), sites, 5, 8) if features is None else features,
        "site_mask": torch.ones(len(wells), sites, dtype=torch.bool),
        "plates": plates,
        "wells": list(wells),
        "pert_info": [
            PerturbationInfo(pert_type=PerturbationType.COMPOUND, broad_sample=f"BRD-{well}")
            for well in wells
        ],
    }


def make_text_cache(broad_samples: list[str], dim: int = 4) -> dict:
    """Build the cache dict ``lookup_text_embeddings`` reads."""
    return {
        "id_to_idx": {sample: i for i, sample in enumerate(broad_samples)},
        "embeddings": torch.arange(len(broad_samples) * dim, dtype=torch.float).reshape(-1, dim),
        "perturbation_ids": list(broad_samples),
    }
