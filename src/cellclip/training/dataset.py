"""Dataset helpers for local CellCLIP training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from benchmark import splits as benchmark_splits_module
from benchmark.data import normalize_subset_label
from cellclip.training.config import (
    CellCLIPDatasetConfig,
    CellCLIPModelConfig,
)
from morphoclip.data.dataset import MorphoCLIPDataset
from morphoclip.data.dataset import collate_fn as morphoclip_collate_fn
from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationInfo, PerturbationType, extract_plate_barcode


@dataclass(slots=True)
class PreparedData:
    """Prepared datasets, loaders, and saved split metadata."""

    dataset: MorphoCLIPDataset
    train_source_dataset: MorphoCLIPDataset
    eval_source_dataset: MorphoCLIPDataset
    train_dataset: Subset
    eval_dataset: Subset
    train_loader: DataLoader
    eval_loader: DataLoader
    tokenizer: PreTrainedTokenizerBase
    smiles_tokenizer: PreTrainedTokenizerBase | None
    split_manifest: pd.DataFrame
    plate_contexts: dict[str, Any]


def resolve_cell_type(plate: str, plate_contexts: dict[str, Any]) -> str:
    """Resolve the annotated cell type for a plate barcode."""
    context = plate_contexts.get(extract_plate_barcode(plate))
    return str(getattr(context, "cell_type", "") or "")


def build_upstream_prompt(
    info: PerturbationInfo,
    cell_type: str,
    *,
    include_smiles: bool = True,
) -> str:
    """Mirror upstream CellCLIP prompt generation."""
    perturbation_type = info.pert_type.value

    if info.pert_type == PerturbationType.COMPOUND:
        target = info.pert_iname or info.broad_sample or "unknown"
        target_info = info.smiles
        prompt = f"{cell_type} cells treated with {perturbation_type}: {target}"
        if include_smiles:
            prompt = f"{prompt}, SMILES: {target_info}"
        if "DMSO" in str(target):
            prompt = f"{cell_type} cells treated with {perturbation_type}: control, {target}"
            if include_smiles:
                prompt = f"{prompt}, SMILES: {target_info}"
        elif "control" in str(target).lower():
            prompt = f"{cell_type} cells treated with {perturbation_type}: control"
        if len(prompt) > 512:
            if include_smiles:
                prompt = f"{cell_type} cells treated with SMILES: {target_info}"
                if len(prompt) > 512:
                    prompt = f"{cell_type} cells treated with {perturbation_type}, {target}"
            else:
                prompt = f"{cell_type} cells treated with {perturbation_type}, {target}"
        return prompt

    if info.pert_type == PerturbationType.CRISPR:
        target = info.gene
        target_info = info.target_sequence
        control_info = info.negcon_control_type or info.control_type
        if "control" in str(control_info).lower():
            return (
                f"{cell_type} cells treated with {perturbation_type} "
                f"sequence: {target_info}, {control_info}"
            )
        if (not str(target_info).strip() or str(target_info).lower() == "nan") and (
            not str(target).strip() or "control" in str(target).lower()
        ):
            return f"{cell_type} cells treated with control, no treatment."
        if "guide" in str(target_info).lower():
            return f"{cell_type} cells treated with {perturbation_type} ,targeting genes: {target}"
        return (
            f"{cell_type} cells treated with {perturbation_type} "
            f"sequence: {target_info}, targeting genes: {target}"
        )

    if info.pert_type == PerturbationType.ORF:
        target = info.gene
        target_info = info.control_type
        control_info = info.negcon_control_type
        if target_info and "control" in str(target_info).lower():
            return (
                f"{cell_type} cells treated with {perturbation_type} "
                f"{target_info} {control_info}, targeting genes: {target}."
            )
        if not str(target).strip() or str(target).lower() == "nan":
            return f"{cell_type} cells treated with control, no treatment."
        return f"{cell_type} cells treated with {perturbation_type}, targeting genes: {target}"

    if info.pert_type in {PerturbationType.NEGCON, PerturbationType.POSCON}:
        return f"{cell_type} cells treated with control, no treatment."

    return f"{cell_type} cells treated with {perturbation_type}: {info.broad_sample}"


@dataclass(slots=True)
class CellCLIPCollator:
    """Picklable collator that pads wells and tokenizes upstream-style prompts."""

    tokenizer: PreTrainedTokenizerBase
    context_length: int
    plate_contexts: dict[str, Any]
    include_smiles_in_prompt: bool = True
    smiles_tokenizer: PreTrainedTokenizerBase | None = None
    chemberta_context_length: int = 512

    def __call__(self, batch: list[Any]) -> dict[str, Any]:
        payload = morphoclip_collate_fn(batch)
        prompts: list[str] = []
        smiles_strings: list[str] = []
        has_smiles: list[bool] = []
        for plate, info in zip(payload["plates"], payload["pert_info"], strict=True):
            cell_type = resolve_cell_type(plate, self.plate_contexts)
            prompts.append(
                build_upstream_prompt(
                    info,
                    cell_type,
                    include_smiles=self.include_smiles_in_prompt,
                )
            )
            smiles = ""
            if info.pert_type == PerturbationType.COMPOUND and str(info.smiles).strip():
                smiles = str(info.smiles).strip()
            smiles_strings.append(smiles)
            has_smiles.append(bool(smiles))
        payload["text"] = prompts
        tokenized = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
            return_tensors="pt",
        )
        payload["text_tokens"] = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if self.smiles_tokenizer is not None:
            smiles_tokenized = self.smiles_tokenizer(
                smiles_strings,
                padding="max_length",
                truncation=True,
                max_length=self.chemberta_context_length,
                return_tensors="pt",
            )
            payload["smiles"] = smiles_strings
            payload["has_smiles"] = torch.tensor(has_smiles, dtype=torch.bool)
            payload["smiles_tokens"] = {
                "input_ids": smiles_tokenized["input_ids"],
                "attention_mask": smiles_tokenized["attention_mask"],
            }
        return payload


def discover_feature_plates(feature_root: Path) -> list[str]:
    """Return available plate directories under a feature cache root."""
    return sorted(path.name for path in feature_root.iterdir() if path.is_dir())


def _resolve_site_cap(
    explicit: int | None,
    fallback: int | None,
) -> int | None:
    """Resolve train/eval site caps with backward-compatible fallback."""
    if explicit is not None:
        return explicit
    return fallback


def _filter_official_contexts(
    dataset: MorphoCLIPDataset,
    official_contexts: dict[tuple[str, str], Any],
) -> None:
    """Restrict a dataset index to wells present in the official split metadata."""
    dataset._index = [  # noqa: SLF001
        entry
        for entry in dataset.index_entries
        if (extract_plate_barcode(entry[0]), entry[1].upper()) in official_contexts
    ]


def build_tokenized_collate_fn(
    tokenizer: PreTrainedTokenizerBase,
    context_length: int,
    plate_contexts: dict[str, Any],
    *,
    include_smiles_in_prompt: bool = True,
    smiles_tokenizer: PreTrainedTokenizerBase | None = None,
    chemberta_context_length: int = 512,
):
    """Wrap the base collate fn with BERT tokenization."""
    return CellCLIPCollator(
        tokenizer=tokenizer,
        context_length=context_length,
        plate_contexts=plate_contexts,
        include_smiles_in_prompt=include_smiles_in_prompt,
        smiles_tokenizer=smiles_tokenizer,
        chemberta_context_length=chemberta_context_length,
    )


def subset_from_manifest(
    dataset: MorphoCLIPDataset,
    manifest: pd.DataFrame,
    subset: str,
    *,
    plate_contexts: dict[str, Any] | None = None,
    unique_perturbations: bool = False,
    max_wells: int | None = None,
) -> Subset:
    """Construct a ``Subset`` from a manifest filtered by subset label."""
    del plate_contexts
    normalized_subset = normalize_subset_label(subset)
    selected = manifest.copy()
    if "subset" in selected.columns:
        selected["subset"] = selected["subset"].map(normalize_subset_label)
        selected = selected[selected["subset"] == normalized_subset]

    keys = {
        (str(row.Metadata_Plate), str(row.Metadata_Well).upper())
        for row in selected.itertuples(index=False)
    }
    indices = [
        idx
        for idx, (plate, well, _) in enumerate(dataset.index_entries)
        if (extract_plate_barcode(plate), well.upper()) in keys
    ]
    if unique_perturbations:
        # Upstream CellCLIP strips the last view token from SAMPLE_KEY when
        # `--unique` is enabled. The local feature cache is already one bag per
        # well, so the upstream-faithful behavior is simply to preserve unique
        # well entries and avoid any extra perturbation-level collapsing.
        indices = list(dict.fromkeys(indices))
    if max_wells is not None:
        indices = indices[:max_wells]
    if not indices:
        raise ValueError(f"No dataset wells matched subset {normalized_subset!r}")
    return Subset(dataset, indices)


def prepare_datasets(
    dataset_config: CellCLIPDatasetConfig,
    model_config: CellCLIPModelConfig,
) -> PreparedData:
    """Build train/eval subsets and data loaders for CellCLIP training."""
    feature_root = Path(dataset_config.feature_root)
    metadata = MetadataIndex.from_config(Path(dataset_config.dataset_config_path))
    plates = discover_feature_plates(feature_root)
    if not plates:
        raise ValueError(f"No plate directories found under {feature_root}")

    dataset = MorphoCLIPDataset(
        feature_dir=feature_root,
        metadata=metadata,
        plates=plates,
        text_level=dataset_config.text_level,
        exclude_controls=dataset_config.exclude_controls,
    )
    official_contexts: dict[tuple[str, str], Any] | None = None
    if dataset_config.split_strategy.startswith("cpjump1_official_"):
        official_contexts = benchmark_splits_module.load_official_split_contexts()
        _filter_official_contexts(dataset, official_contexts)

    plate_contexts = benchmark_splits_module.load_plate_contexts()
    if dataset_config.split_manifest_path:
        manifest = pd.read_csv(dataset_config.split_manifest_path)
    else:
        manifest = benchmark_splits_module.build_split_manifest(
            dataset,
            strategy=dataset_config.split_strategy,
            val_fraction=dataset_config.val_fraction,
            test_fraction=dataset_config.test_fraction,
            seed=dataset_config.seed,
        )

    train_source_dataset = MorphoCLIPDataset(
        feature_dir=feature_root,
        metadata=metadata,
        plates=plates,
        text_level=dataset_config.text_level,
        exclude_controls=dataset_config.exclude_controls,
        max_sites_per_well=_resolve_site_cap(
            dataset_config.train_max_sites_per_well,
            dataset_config.max_sites_per_well,
        ),
    )
    eval_source_dataset = MorphoCLIPDataset(
        feature_dir=feature_root,
        metadata=metadata,
        plates=plates,
        text_level=dataset_config.text_level,
        exclude_controls=dataset_config.exclude_controls,
        max_sites_per_well=_resolve_site_cap(
            dataset_config.eval_max_sites_per_well,
            dataset_config.max_sites_per_well,
        ),
    )
    if official_contexts is not None:
        _filter_official_contexts(train_source_dataset, official_contexts)
        _filter_official_contexts(eval_source_dataset, official_contexts)

    train_dataset = subset_from_manifest(
        train_source_dataset,
        manifest,
        dataset_config.subset,
        plate_contexts=plate_contexts,
        unique_perturbations=dataset_config.unique_perturbations,
        max_wells=dataset_config.max_train_wells,
    )
    eval_dataset = subset_from_manifest(
        eval_source_dataset,
        manifest,
        dataset_config.eval_subset,
        plate_contexts=plate_contexts,
        unique_perturbations=dataset_config.unique_perturbations,
        max_wells=dataset_config.max_eval_wells,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_config.tokenizer_name)
    smiles_tokenizer = None
    include_smiles_in_prompt = True
    if model_config.variant in {"chemberta_film", "chemberta"}:
        smiles_tokenizer = AutoTokenizer.from_pretrained(model_config.chemberta_tokenizer_name)
        include_smiles_in_prompt = model_config.chem_prompt_policy == "keep_smiles"
    collate = build_tokenized_collate_fn(
        tokenizer,
        model_config.context_length,
        plate_contexts=plate_contexts,
        include_smiles_in_prompt=include_smiles_in_prompt,
        smiles_tokenizer=smiles_tokenizer,
        chemberta_context_length=model_config.chemberta_context_length,
    )

    from morphoclip.utils.device import resolve_num_workers

    num_workers = resolve_num_workers(dataset_config.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataset_config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dataset_config.pin_memory,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=dataset_config.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dataset_config.pin_memory,
        collate_fn=collate,
    )

    return PreparedData(
        dataset=dataset,
        train_source_dataset=train_source_dataset,
        eval_source_dataset=eval_source_dataset,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        smiles_tokenizer=smiles_tokenizer,
        split_manifest=manifest,
        plate_contexts=plate_contexts,
    )
