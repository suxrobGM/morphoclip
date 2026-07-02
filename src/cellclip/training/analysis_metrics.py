"""Retrieval and PCA metrics for CellCLIP run analysis."""

import torch


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
