"""Reusable post-training analysis for local CellCLIP runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

import benchmark.splits as benchmark_splits_module
from cellclip.training.config import load_training_config
from cellclip.training.dataset import (
    build_upstream_prompt,
    prepare_datasets,
    resolve_cell_type,
)
from cellclip.training.engine import resolve_device
from cellclip.training.model import build_cellclip_model
from morphoclip.data.perturbation import extract_plate_barcode


def to_serializable(value: Any) -> Any:
    """Recursively convert analysis values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def resolve_checkpoint_path(run_dir: Path) -> Path:
    """Resolve the preferred checkpoint for a training run."""
    best = run_dir / "checkpoints" / "best.pt"
    if best.exists():
        return best
    last = run_dir / "checkpoints" / "last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No CellCLIP checkpoint found under {run_dir / 'checkpoints'}")


def compute_pca_stats(features: torch.Tensor) -> dict[str, float]:
    """Summarize anisotropy using PCA energy concentration."""
    if features.numel() == 0:
        return {"samples": 0, "top1_fraction": 0.0, "top10_fraction": 0.0, "mean_norm": 0.0}
    if features.shape[0] < 2:
        return {
            "samples": int(features.shape[0]),
            "top1_fraction": 1.0,
            "top10_fraction": 1.0,
            "mean_norm": float(features.float().norm(dim=1).mean().item()),
        }
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    energy = singular_values.square()
    fractions = energy / energy.sum().clamp(min=torch.finfo(energy.dtype).eps)
    return {
        "samples": int(features.shape[0]),
        "top1_fraction": float(fractions[0].item()),
        "top10_fraction": float(fractions[:10].sum().item()),
        "mean_norm": float(features.float().norm(dim=1).mean().item()),
    }


def compute_grouped_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    prompts: list[str],
    broad_samples: list[str],
) -> dict[str, float]:
    """Compute exact and grouped retrieval on eval embeddings."""
    if image_features.numel() == 0:
        return {"exact_R@1": 0.0, "prompt_R@1": 0.0, "broad_sample_R@1": 0.0}
    logits = image_features @ text_features.t()
    order = torch.argsort(logits, dim=1, descending=True)
    size = logits.shape[0]
    indices = torch.arange(size)
    prompt_labels = prompts
    broad_labels = broad_samples
    metrics: dict[str, float] = {}
    top1 = order[:, 0]
    metrics["exact_R@1"] = float((top1 == indices).float().mean().item())
    metrics["prompt_R@1"] = float(
        sum(prompt_labels[int(j)] == prompt_labels[i] for i, j in enumerate(top1.tolist())) / size
    )
    metrics["broad_sample_R@1"] = float(
        sum(broad_labels[int(j)] == broad_labels[i] for i, j in enumerate(top1.tolist())) / size
    )
    return metrics


def compute_perturbation_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    prompts: list[str],
    broad_samples: list[str],
    pert_types: list[str],
) -> dict[str, dict[str, float]]:
    """Compute grouped retrieval stratified by perturbation type."""
    metrics: dict[str, dict[str, float]] = {}
    for pert_type in sorted(set(pert_types)):
        indices = [idx for idx, label in enumerate(pert_types) if label == pert_type]
        if not indices:
            continue
        metrics[pert_type] = compute_grouped_retrieval_metrics(
            image_features[indices],
            text_features[indices],
            [prompts[idx] for idx in indices],
            [broad_samples[idx] for idx in indices],
        )
    return metrics


def compute_split_pca_stats(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    pert_types: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute PCA stats for compound and non-compound slices."""
    slices = {
        "compound": [idx for idx, label in enumerate(pert_types) if label == "compound"],
        "non_compound": [idx for idx, label in enumerate(pert_types) if label != "compound"],
    }
    stats: dict[str, dict[str, dict[str, float]]] = {}
    for name, indices in slices.items():
        stats[name] = {
            "image": compute_pca_stats(image_features[indices]),
            "text": compute_pca_stats(text_features[indices]),
        }
    return stats


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


def load_benchmark_tables(benchmark_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load benchmark summary tables when available."""
    tables_dir = benchmark_dir / "tables"
    if not tables_dir.exists():
        return {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for csv_path in sorted(tables_dir.glob("*summary.csv")):
        frame = pd.read_csv(csv_path)
        tables[csv_path.stem] = frame.to_dict(orient="records")
    return tables


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


def build_comparison(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    """Build a compact comparison between two run summaries."""
    comparison: dict[str, Any] = {}
    for key in (
        "eval_retrieval",
        "compound_eval_retrieval",
        "image_pca",
        "text_pca",
        "fusion_diagnostics",
    ):
        comparison[key] = {}
        for metric_name, primary_value in primary.get(key, {}).items():
            secondary_value = secondary.get(key, {}).get(metric_name)
            if secondary_value is None:
                continue
            comparison[key][metric_name] = {
                "primary": primary_value,
                "secondary": secondary_value,
                "delta": primary_value - secondary_value,
            }
    return comparison


def render_report(
    primary: dict[str, Any],
    *,
    secondary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> str:
    """Render a human-readable Markdown report."""
    lines = [
        "# CellCLIP Run Analysis",
        "",
        f"- Run: `{primary['run_dir']}`",
        f"- Checkpoint: `{primary['checkpoint_path']}`",
        "",
        "## Eval Retrieval",
    ]
    for key, value in primary["eval_retrieval"].items():
        lines.append(f"- {key}: {value:.6f}")
    duplicate_train = json.dumps(
        to_serializable(primary["duplicate_stats"]["train"]), sort_keys=True
    )
    duplicate_eval = json.dumps(to_serializable(primary["duplicate_stats"]["eval"]), sort_keys=True)
    fusion_diagnostics = json.dumps(to_serializable(primary["fusion_diagnostics"]), sort_keys=True)
    lines.extend(
        [
            "",
            "## Compound Retrieval",
            *(f"- {key}: {value:.6f}" for key, value in primary["compound_eval_retrieval"].items()),
            "",
            "## Perturbation Retrieval",
            *(
                f"- {name}: {json.dumps(to_serializable(metrics), sort_keys=True)}"
                for name, metrics in primary["perturbation_retrieval"].items()
            ),
            "",
            "## Duplicate Stats",
            f"- Train: {duplicate_train}",
            f"- Eval: {duplicate_eval}",
            "",
            "## Chem Diagnostics",
            f"- Has SMILES fraction: {primary['has_smiles_fraction']:.6f}",
            f"- Fusion: {fusion_diagnostics}",
            "",
            "## PCA Diagnostics",
            f"- Image: {json.dumps(to_serializable(primary['image_pca']), sort_keys=True)}",
            f"- Text: {json.dumps(to_serializable(primary['text_pca']), sort_keys=True)}",
            f"- Split: {json.dumps(to_serializable(primary['split_pca']), sort_keys=True)}",
        ]
    )
    if primary.get("benchmark_tables"):
        lines.extend(["", "## Benchmark Tables"])
        for name, rows in primary["benchmark_tables"].items():
            lines.append(f"- {name}: {len(rows)} rows")
    if secondary is not None and comparison is not None:
        lines.extend(["", "## Comparison", f"- Compare run: `{secondary['run_dir']}`"])
        for section, metrics in comparison.items():
            lines.append(f"- {section}: {json.dumps(to_serializable(metrics), sort_keys=True)}")
    return "\n".join(lines) + "\n"


def write_analysis_outputs(
    output_dir: Path,
    primary: dict[str, Any],
    *,
    secondary: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write analysis outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    payload = {"primary": primary}
    if secondary is not None:
        payload["secondary"] = secondary
    if comparison is not None:
        payload["comparison"] = comparison
    summary_path.write_text(
        json.dumps(to_serializable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        render_report(primary, secondary=secondary, comparison=comparison),
        encoding="utf-8",
    )
    return summary_path, report_path
