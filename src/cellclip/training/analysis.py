"""Reusable post-training analysis for local CellCLIP runs.

Pure retrieval/PCA metrics live in :mod:`cellclip.training.analysis_metrics` and
serialization/reporting in :mod:`cellclip.training.analysis_report`; this module
owns the model/checkpoint-heavy embedding collection and the top-level
:func:`build_run_summary`, and re-exports the moved helpers for existing callers.
"""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

import benchmark.splits as benchmark_splits_module
from cellclip.training.analysis_metrics import (
    compute_grouped_retrieval_metrics,
    compute_pca_stats,
    compute_perturbation_retrieval_metrics,
    compute_split_pca_stats,
)
from cellclip.training.analysis_report import (
    build_comparison,
    load_benchmark_tables,
    render_report,
    to_serializable,
    write_analysis_outputs,
)
from cellclip.training.config import load_training_config
from cellclip.training.dataset import (
    build_upstream_prompt,
    prepare_datasets,
    resolve_cell_type,
)
from cellclip.training.engine import resolve_device
from cellclip.training.model import build_cellclip_model
from morphoclip.data.perturbation import extract_plate_barcode

__all__ = [
    "to_serializable",
    "compute_pca_stats",
    "compute_grouped_retrieval_metrics",
    "compute_perturbation_retrieval_metrics",
    "compute_split_pca_stats",
    "load_benchmark_tables",
    "build_comparison",
    "render_report",
    "write_analysis_outputs",
    "resolve_checkpoint_path",
    "summarize_subset_duplicates",
    "collect_eval_embeddings",
    "build_run_summary",
]


def resolve_checkpoint_path(run_dir: Path) -> Path:
    """Resolve the preferred checkpoint for a training run."""
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        return best
    last = run_dir / "checkpoints" / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No CellCLIP checkpoint found under {run_dir / 'checkpoints'}")


def summarize_subset_duplicates(
    prepared,
    subset_name: str,
    *,
    include_smiles_in_prompt: bool,
) -> dict[str, float | int]:
    """Summarize duplicated prompts and perturbation keys for one subset."""
    subset = prepared.train_dataset if subset_name == "train" else prepared.eval_dataset
    plate_contexts = benchmark_splits_module.load_plate_contexts()
    prompt_counts: dict[str, int] = {}
    plate_well_counts: dict[tuple[str, str], int] = {}
    cell_broad_counts: dict[tuple[str, str], int] = {}
    cell_time_pert_broad_counts: dict[tuple[str, int, str, str], int] = {}
    for idx in subset.indices:
        plate, well, _ = prepared.dataset.index_entries[idx]
        barcode = extract_plate_barcode(plate)
        info = prepared.dataset.metadata.lookup(barcode, well)
        context = plate_contexts.get(barcode)
        cell_type = resolve_cell_type(plate, plate_contexts)
        prompt = build_upstream_prompt(
            info,
            cell_type,
            include_smiles=include_smiles_in_prompt,
        )
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        plate_key = (barcode, well.upper())
        plate_well_counts[plate_key] = plate_well_counts.get(plate_key, 0) + 1
        broad_sample = str(info.broad_sample).strip()
        if broad_sample:
            cell_broad_key = (cell_type, broad_sample)
            cell_broad_counts[cell_broad_key] = cell_broad_counts.get(cell_broad_key, 0) + 1
            if context is not None:
                slice_key = (
                    cell_type,
                    int(context.timepoint),
                    context.perturbation,
                    broad_sample,
                )
                cell_time_pert_broad_counts[slice_key] = (
                    cell_time_pert_broad_counts.get(slice_key, 0) + 1
                )
    total = len(subset)
    prompt_values = list(prompt_counts.values())
    plate_values = list(plate_well_counts.values())
    cell_broad_values = list(cell_broad_counts.values())
    cell_time_values = list(cell_time_pert_broad_counts.values())
    return {
        "samples": total,
        "unique_plate_wells": len(plate_well_counts),
        "plate_well_duplicate_instances": int(sum(v - 1 for v in plate_values if v > 1)),
        "unique_prompts": len(prompt_counts),
        "prompt_duplicate_instances": int(sum(v - 1 for v in prompt_values if v > 1)),
        "unique_cell_broad_samples": len(cell_broad_counts),
        "cell_broad_duplicate_instances": int(sum(v - 1 for v in cell_broad_values if v > 1)),
        "unique_cell_time_pert_broad_samples": len(cell_time_pert_broad_counts),
        "cell_time_pert_broad_duplicate_instances": int(
            sum(v - 1 for v in cell_time_values if v > 1)
        ),
    }


def collect_eval_embeddings(
    run_dir: Path,
    *,
    max_eval_wells: int | None = None,
) -> dict[str, Any]:
    """Collect eval-set embeddings and labels for a run."""
    config = load_training_config(run_dir / "resolved_config.yaml")
    config.dataset.num_workers = 0
    if max_eval_wells is not None:
        config.dataset.max_eval_wells = max_eval_wells
    prepared = prepare_datasets(config.dataset, config.model)
    device = resolve_device(config.runtime.device)
    model = build_cellclip_model(config.model).to(device)
    checkpoint = torch.load(resolve_checkpoint_path(run_dir), map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []
    prompts: list[str] = []
    broad_samples: list[str] = []
    pert_types: list[str] = []
    has_smiles_count = 0
    total_samples = 0
    fusion_totals: dict[str, float] = {}
    fusion_counts: dict[str, int] = {}

    with torch.no_grad():
        for batch in prepared.eval_loader:
            features = batch["features"].to(device, non_blocking=True)
            text_tokens = {
                key: value.to(device, non_blocking=True)
                for key, value in batch["text_tokens"].items()
            }
            smiles_tokens = None
            if batch.get("smiles_tokens") is not None:
                smiles_tokens = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch["smiles_tokens"].items()
                }
            has_smiles = batch.get("has_smiles")
            if has_smiles is not None:
                has_smiles = has_smiles.to(device, non_blocking=True)
            pooled_images = model.encode_mil(features)
            image_features = F.normalize(model.encode_image(pooled_images), dim=1)
            if hasattr(model, "encode_text_with_diagnostics"):
                text_hidden, batch_diagnostics = model.encode_text_with_diagnostics(
                    text_tokens,
                    smiles=smiles_tokens,
                    has_smiles=has_smiles,
                )
            else:
                text_hidden = model.encode_text(
                    text_tokens,
                    smiles=smiles_tokens,
                    has_smiles=has_smiles,
                )
                batch_diagnostics = {}
            text_features = F.normalize(text_hidden, dim=1)
            image_batches.append(image_features.detach().cpu())
            text_batches.append(text_features.detach().cpu())
            prompts.extend(batch["text"])
            broad_samples.extend(str(info.broad_sample).strip() for info in batch["pert_info"])
            pert_types.extend(info.pert_type.value for info in batch["pert_info"])
            batch_size = len(batch["pert_info"])
            total_samples += batch_size
            smiles_in_batch = int(has_smiles.sum().item()) if has_smiles is not None else 0
            has_smiles_count += smiles_in_batch
            if smiles_in_batch > 0:
                for key, value in batch_diagnostics.items():
                    fusion_totals[key] = fusion_totals.get(key, 0.0) + (value * smiles_in_batch)
                    fusion_counts[key] = fusion_counts.get(key, 0) + smiles_in_batch

    include_smiles = not (
        config.model.variant in {"chemberta_film", "chemberta"}
        and config.model.chem_prompt_policy == "remove_smiles"
    )
    return {
        "config": config,
        "prepared": prepared,
        "image_features": torch.cat(image_batches, dim=0),
        "text_features": torch.cat(text_batches, dim=0),
        "prompts": prompts,
        "broad_samples": broad_samples,
        "pert_types": pert_types,
        "has_smiles_fraction": (has_smiles_count / total_samples) if total_samples else 0.0,
        "fusion_diagnostics": {
            key: fusion_totals[key] / fusion_counts[key] for key in sorted(fusion_totals)
        },
        "include_smiles_in_prompt": include_smiles,
    }


def build_run_summary(
    run_dir: Path,
    *,
    max_eval_wells: int | None = None,
    compare_benchmark_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a reusable analysis summary for a CellCLIP run."""
    run_dir = run_dir.resolve()
    metrics_path = run_dir / "metrics.csv"
    metrics_rows = []
    final_metrics: dict[str, Any] = {}
    if metrics_path.exists():
        metrics_frame = pd.read_csv(metrics_path)
        metrics_rows = metrics_frame.to_dict(orient="records")
        if not metrics_frame.empty:
            final_metrics = metrics_frame.iloc[-1].to_dict()

    collected = collect_eval_embeddings(run_dir, max_eval_wells=max_eval_wells)
    image_features = collected["image_features"]
    text_features = collected["text_features"]
    grouped_metrics = compute_grouped_retrieval_metrics(
        image_features,
        text_features,
        collected["prompts"],
        collected["broad_samples"],
    )
    perturbation_metrics = compute_perturbation_retrieval_metrics(
        image_features,
        text_features,
        collected["prompts"],
        collected["broad_samples"],
        collected["pert_types"],
    )
    compound_indices = [
        idx for idx, label in enumerate(collected["pert_types"]) if label == "compound"
    ]
    duplicate_stats = {
        "train": summarize_subset_duplicates(
            collected["prepared"],
            "train",
            include_smiles_in_prompt=collected["include_smiles_in_prompt"],
        ),
        "eval": summarize_subset_duplicates(
            collected["prepared"],
            "eval",
            include_smiles_in_prompt=collected["include_smiles_in_prompt"],
        ),
    }

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(resolve_checkpoint_path(run_dir)),
        "metrics_history": metrics_rows,
        "final_metrics": final_metrics,
        "eval_retrieval": grouped_metrics,
        "perturbation_retrieval": perturbation_metrics,
        "duplicate_stats": duplicate_stats,
        "has_smiles_fraction": collected["has_smiles_fraction"],
        "fusion_diagnostics": collected["fusion_diagnostics"],
        "image_pca": compute_pca_stats(image_features),
        "text_pca": compute_pca_stats(text_features),
        "split_pca": compute_split_pca_stats(
            image_features,
            text_features,
            collected["pert_types"],
        ),
    }
    summary["compound_eval_retrieval"] = compute_grouped_retrieval_metrics(
        image_features[compound_indices],
        text_features[compound_indices],
        [collected["prompts"][idx] for idx in compound_indices],
        [collected["broad_samples"][idx] for idx in compound_indices],
    )
    if compare_benchmark_dir is not None:
        summary["benchmark_tables"] = load_benchmark_tables(compare_benchmark_dir)
    return summary
