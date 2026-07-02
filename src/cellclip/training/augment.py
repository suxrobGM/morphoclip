"""Training-only feature-space augmentations for CellCLIP."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.perturbation import PerturbationInfo, extract_plate_barcode


@dataclass(frozen=True, slots=True)
class PerturbationSliceKey:
    """Key for slice-aware same-perturbation partner lookup."""

    cell_type: str
    perturbation: str
    timepoint: int
    broad_sample: str


class FeatureBagAugmenter:
    """Apply site-level feature augmentations during training only."""

    def __init__(
        self,
        *,
        dataset: MorphoCLIPDataset,
        train_indices: list[int],
        plate_contexts: dict[str, Any],
        within_well_interp_sites: int,
        same_pert_interp_sites: int,
        interp_alpha: float,
    ) -> None:
        self._dataset = dataset
        self._plate_contexts = plate_contexts
        self._within_well_interp_sites = within_well_interp_sites
        self._same_pert_interp_sites = same_pert_interp_sites
        self._beta = torch.distributions.Beta(interp_alpha, interp_alpha)
        self._partner_indices: dict[PerturbationSliceKey, list[int]] = defaultdict(list)
        self._plate_well_to_index: dict[tuple[str, str], int] = {}
        self._partner_feature_cache: dict[int, torch.Tensor] = {}

        for index in train_indices:
            plate, well, _ = dataset.index_entries[index]
            barcode = extract_plate_barcode(plate)
            info = dataset.metadata.lookup(barcode, well)
            context = plate_contexts.get(barcode)
            if context is None or not info.broad_sample:
                continue
            key = PerturbationSliceKey(
                cell_type=str(getattr(context, "cell_type", "") or ""),
                perturbation=info.pert_type.value,
                timepoint=int(getattr(context, "timepoint", 0) or 0),
                broad_sample=str(info.broad_sample),
            )
            self._partner_indices[key].append(index)
            self._plate_well_to_index[(plate, well.upper())] = index

    @property
    def enabled(self) -> bool:
        """Whether any augmentation is active."""
        return self._within_well_interp_sites > 0 or self._same_pert_interp_sites > 0

    def _load_partner_features(self, index: int) -> torch.Tensor:
        if index not in self._partner_feature_cache:
            self._partner_feature_cache[index] = self._dataset[index].features
        return self._partner_feature_cache[index]

    def _sample_lambda(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._beta.sample().to(device=device, dtype=dtype)

    @staticmethod
    def _select_positions(
        valid_indices: torch.Tensor,
        count: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if count <= 0 or valid_indices.numel() == 0:
            return valid_indices.new_empty((0,), dtype=torch.long)
        limit = min(count, int(valid_indices.numel()))
        order = torch.randperm(valid_indices.numel(), device=device)[:limit]
        return valid_indices[order]

    def _apply_within_well(
        self,
        augmented: torch.Tensor,
        batch_index: int,
        valid_indices: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        positions = self._select_positions(
            valid_indices,
            self._within_well_interp_sites,
            device=device,
        )
        if positions.numel() == 0 or valid_indices.numel() < 2:
            return positions

        for position in positions.tolist():
            donors = valid_indices[torch.randperm(valid_indices.numel(), device=device)[:2]]
            lam = self._sample_lambda(device=device, dtype=augmented.dtype)
            augmented[batch_index, position] = (
                lam * augmented[batch_index, donors[0]]
                + (1 - lam) * augmented[batch_index, donors[1]]
            )
        return positions

    def _resolve_partner_index(
        self,
        *,
        plate: str,
        well: str,
        info: PerturbationInfo,
    ) -> int | None:
        barcode = extract_plate_barcode(plate)
        context = self._plate_contexts.get(barcode)
        if context is None or not info.broad_sample:
            return None

        key = PerturbationSliceKey(
            cell_type=str(getattr(context, "cell_type", "") or ""),
            perturbation=info.pert_type.value,
            timepoint=int(getattr(context, "timepoint", 0) or 0),
            broad_sample=str(info.broad_sample),
        )
        candidates = self._partner_indices.get(key, [])
        if not candidates:
            return None

        current = self._plate_well_to_index.get((plate, well.upper()))
        preferred: list[int] = []
        fallback: list[int] = []
        for candidate in candidates:
            if candidate == current:
                continue
            candidate_plate, _, _ = self._dataset.index_entries[candidate]
            if candidate_plate != plate:
                preferred.append(candidate)
            else:
                fallback.append(candidate)
        pool = preferred or fallback
        if not pool:
            return None
        choice = torch.randint(len(pool), (1,)).item()
        return pool[choice]

    def _apply_same_perturbation(
        self,
        augmented: torch.Tensor,
        batch_index: int,
        valid_indices: torch.Tensor,
        *,
        device: torch.device,
        plate: str,
        well: str,
        info: PerturbationInfo,
        reserved: torch.Tensor,
    ) -> None:
        remaining = valid_indices
        if reserved.numel() > 0:
            remaining = valid_indices[~torch.isin(valid_indices, reserved)]
        positions = self._select_positions(
            remaining,
            self._same_pert_interp_sites,
            device=device,
        )
        if positions.numel() == 0:
            return

        partner_index = self._resolve_partner_index(plate=plate, well=well, info=info)
        if partner_index is None:
            return
        partner = self._load_partner_features(partner_index).to(
            device=device,
            dtype=augmented.dtype,
        )
        partner_valid = torch.arange(partner.shape[0], device=device)
        if partner_valid.numel() == 0:
            return

        for position in positions.tolist():
            self_choice = valid_indices[torch.randint(valid_indices.numel(), (1,), device=device)][
                0
            ]
            partner_choice = partner_valid[
                torch.randint(partner_valid.numel(), (1,), device=device)
            ][0]
            lam = self._sample_lambda(device=device, dtype=augmented.dtype)
            augmented[batch_index, position] = (
                lam * augmented[batch_index, self_choice] + (1 - lam) * partner[partner_choice]
            )

    def __call__(
        self,
        features: torch.Tensor,
        *,
        site_mask: torch.Tensor | None,
        plates: list[str],
        wells: list[str],
        pert_info: list[PerturbationInfo],
    ) -> torch.Tensor:
        if not self.enabled:
            return features

        augmented = features.clone()
        device = augmented.device
        for batch_index, info in enumerate(pert_info):
            if site_mask is None:
                valid_indices = torch.arange(augmented.shape[1], device=device)
            else:
                valid_indices = torch.nonzero(site_mask[batch_index], as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                continue

            reserved = self._apply_within_well(
                augmented,
                batch_index,
                valid_indices,
                device=device,
            )
            self._apply_same_perturbation(
                augmented,
                batch_index,
                valid_indices,
                device=device,
                plate=plates[batch_index],
                well=wells[batch_index],
                info=info,
                reserved=reserved,
            )
        return augmented
