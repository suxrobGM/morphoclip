"""Diagnose DINOv3 feature extraction quality.

Loads extracted CLS tokens, computes quantitative metrics, saves a JSON
report, and generates diagnostic plots as PNGs.

Outputs saved to ``data/visualizations/{plate}/``:
- ``metrics.json``           — all quantitative measurements
- ``similarity_heatmap.png`` — inter-channel cosine similarity matrix
- ``norm_distributions.png`` — L2 norm histograms per channel
- ``pca_channels.png``       — 2D PCA scatter colored by channel
- ``pca_variance.png``       — PCA cumulative explained variance curve
- ``variance_decomp.png``    — between-channel / between-site / residual

Usage:
    uv run python scripts/features/diagnose_features.py
    uv run python scripts/features/diagnose_features.py --plate BR00116991
    uv run python scripts/features/diagnose_features.py --max-sites 500
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morphoclip.data.image_loader import CHANNEL_NAMES, FLUORESCENCE_CHANNELS  # noqa: E402

console = Console()

# Short labels for plots
CH_LABELS: list[str] = [CHANNEL_NAMES[ch] for ch in FLUORESCENCE_CHANNELS]
CH_SHORT: list[str] = ["Mito", "Actin", "Golgi/PM", "ER", "DNA"]


# ── helpers ──────────────────────────────────────────────────────────────────


def load_features(
    feature_dir: Path,
    *,
    max_sites: int = 500,
    seed: int = 42,
) -> tuple[torch.Tensor, list[str]]:
    """Load feature .pt files, randomly sampling if needed.

    Returns:
        ``(features, names)`` — features shape ``(N, 5, D)``.
    """
    pt_files = sorted(feature_dir.glob("*.pt"))
    if len(pt_files) > max_sites:
        random.seed(seed)
        pt_files = sorted(random.sample(pt_files, max_sites))

    features = []
    names = []
    for f in pt_files:
        features.append(torch.load(f, weights_only=True))
        names.append(f.stem)

    return torch.stack(features), names


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity along last dim."""
    a_n = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_n = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return (a_n * b_n).sum(dim=-1)


# ── metric computation ───────────────────────────────────────────────────────


def compute_metrics(features: torch.Tensor, *, seed: int = 42) -> dict:
    """Compute all diagnostic metrics from a ``(N, 5, D)`` tensor.

    Returns a JSON-serializable dict with all measurements.
    """
    n_sites, n_channels, dim = features.shape
    m: dict = {
        "n_sites": n_sites,
        "n_channels": n_channels,
        "feature_dim": dim,
    }

    # 1. Per-channel statistics
    ch_stats = {}
    for i, ch in enumerate(FLUORESCENCE_CHANNELS):
        ch_feats = features[:, i, :]
        norms = ch_feats.norm(dim=-1)
        ch_stats[CHANNEL_NAMES[ch]] = {
            "l2_norm_mean": round(norms.mean().item(), 4),
            "l2_norm_std": round(norms.std().item(), 4),
            "mean": round(ch_feats.mean().item(), 6),
            "std": round(ch_feats.std().item(), 4),
            "min": round(ch_feats.min().item(), 4),
            "max": round(ch_feats.max().item(), 4),
        }
    m["channel_statistics"] = ch_stats

    # 2. Intra-site inter-channel cosine similarity
    sim_matrix = torch.zeros(n_channels, n_channels)
    sim_std_matrix = torch.zeros(n_channels, n_channels)
    for i in range(n_channels):
        for j in range(n_channels):
            sims = cosine_sim(features[:, i, :], features[:, j, :])
            sim_matrix[i, j] = sims.mean()
            sim_std_matrix[i, j] = sims.std()

    m["intra_site_similarity"] = {
        "matrix": [
            [round(sim_matrix[i, j].item(), 4) for j in range(n_channels)]
            for i in range(n_channels)
        ],
        "matrix_std": [
            [round(sim_std_matrix[i, j].item(), 4) for j in range(n_channels)]
            for i in range(n_channels)
        ],
        "channel_labels": CH_LABELS,
    }

    off_diag = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            off_diag.append(sim_matrix[i, j].item())
    m["intra_site_similarity"]["off_diagonal_mean"] = round(float(np.mean(off_diag)), 4)
    m["intra_site_similarity"]["off_diagonal_std"] = round(float(np.std(off_diag)), 4)

    # 3. Cross-site same-channel vs cross-channel
    n_pairs = min(5000, n_sites * (n_sites - 1) // 2)
    random.seed(seed)
    idx_a = torch.tensor([random.randint(0, n_sites - 1) for _ in range(n_pairs)])
    idx_b = torch.tensor([random.randint(0, n_sites - 1) for _ in range(n_pairs)])
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    same_ch = {}
    for i, ch in enumerate(FLUORESCENCE_CHANNELS):
        sims = cosine_sim(features[idx_a, i], features[idx_b, i])
        same_ch[CHANNEL_NAMES[ch]] = round(sims.mean().item(), 4)

    cross_ch_vals = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            sims = cosine_sim(features[idx_a, i], features[idx_b, j])
            cross_ch_vals.append(sims.mean().item())

    same_mean = float(np.mean(list(same_ch.values())))
    cross_mean = float(np.mean(cross_ch_vals))
    m["cross_site_similarity"] = {
        "same_channel": same_ch,
        "same_channel_mean": round(same_mean, 4),
        "cross_channel_mean": round(cross_mean, 4),
        "gap": round(same_mean - cross_mean, 4),
    }

    # 4. Variance decomposition
    grand_mean = features.mean(dim=(0, 1))
    channel_means = features.mean(dim=0)
    site_means = features.mean(dim=1)
    total_var = ((features - grand_mean) ** 2).mean().item()
    between_ch_var = ((channel_means - grand_mean) ** 2).mean().item()
    between_site_var = ((site_means - grand_mean) ** 2).mean().item()
    residual_var = total_var - between_ch_var - between_site_var

    m["variance_decomposition"] = {
        "total": round(total_var, 6),
        "between_channel": round(between_ch_var, 6),
        "between_channel_pct": round(100 * between_ch_var / total_var, 1),
        "between_site": round(between_site_var, 6),
        "between_site_pct": round(100 * between_site_var / total_var, 1),
        "residual": round(residual_var, 6),
        "residual_pct": round(100 * residual_var / total_var, 1),
    }

    # 5. PCA explained variance
    flat = features.reshape(-1, dim).float()
    flat_centered = flat - flat.mean(dim=0)
    if flat_centered.shape[0] > 2000:
        random.seed(seed)
        idx = random.sample(range(flat_centered.shape[0]), 2000)
        flat_centered = flat_centered[idx]

    _, s, _ = torch.svd_lowrank(flat_centered, q=50)
    explained = (s**2) / (s**2).sum()
    cumulative = explained.cumsum(dim=0)

    pca_components = []
    for i in range(min(50, len(explained))):
        pca_components.append(
            {
                "pc": i + 1,
                "variance_explained": round(100 * explained[i].item(), 2),
                "cumulative": round(100 * cumulative[i].item(), 2),
            }
        )

    n_90 = None
    for i, c in enumerate(cumulative):
        if c >= 0.90:
            n_90 = i + 1
            break

    m["pca"] = {
        "components": pca_components,
        "n_components_90pct": n_90,
    }

    # Summary verdicts
    verdicts = []
    off_mean = m["intra_site_similarity"]["off_diagonal_mean"]
    if off_mean > 0.95:
        verdicts.append("FAIL: channels nearly identical (cosine > 0.95)")
    elif off_mean > 0.85:
        verdicts.append("WARN: moderate channel similarity (0.85-0.95)")
    else:
        verdicts.append("PASS: good channel separation (cosine < 0.85)")

    if m["cross_site_similarity"]["gap"] > 0.02:
        verdicts.append("PASS: channels distinguishable across sites")
    elif m["cross_site_similarity"]["gap"] > 0:
        verdicts.append("WARN: weak cross-site channel signal")
    else:
        verdicts.append("FAIL: no cross-site channel signal")

    if m["variance_decomposition"]["between_channel_pct"] > 5:
        verdicts.append("PASS: channel identity explains >5% of variance")
    else:
        verdicts.append("WARN: channel identity explains <5% of variance")

    m["verdicts"] = verdicts
    return m


# ── plot generation ──────────────────────────────────────────────────────────


def plot_similarity_heatmap(metrics: dict, output_dir: Path) -> Path:
    """Save inter-channel cosine similarity heatmap."""
    matrix = np.array(metrics["intra_site_similarity"]["matrix"])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(CH_SHORT)))
    ax.set_xticklabels(CH_SHORT, rotation=45, ha="right")
    ax.set_yticks(range(len(CH_SHORT)))
    ax.set_yticklabels(CH_SHORT)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title("Intra-site Inter-channel Cosine Similarity")
    fig.tight_layout()

    out = output_dir / "similarity_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_norm_distributions(features: torch.Tensor, output_dir: Path) -> Path:
    """Save L2 norm histograms per channel."""
    n_channels = features.shape[1]
    colors = ["#c44e52", "#dd8452", "#55a868", "#ccb974", "#4c72b0"]

    fig, axes = plt.subplots(1, n_channels, figsize=(3.2 * n_channels, 3), sharey=True)
    for i, (_ch, ax) in enumerate(zip(FLUORESCENCE_CHANNELS, axes, strict=True)):
        norms = features[:, i, :].norm(dim=-1).numpy()
        ax.hist(norms, bins=40, color=colors[i], edgecolor="white", alpha=0.85)
        ax.set_title(CH_SHORT[i], fontsize=10)
        ax.set_xlabel("L2 Norm")
        ax.axvline(
            norms.mean(),
            color="black",
            linestyle="--",
            linewidth=1,
            label=f"mean={norms.mean():.1f}",
        )
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Count")
    fig.suptitle("Feature L2 Norm Distributions", fontsize=12)
    fig.tight_layout()

    out = output_dir / "norm_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pca_channels(features: torch.Tensor, output_dir: Path, *, seed: int = 42) -> Path:
    """Save 2D PCA scatter plot colored by channel."""
    n_sites, n_channels, dim = features.shape
    flat = features.reshape(-1, dim).float()
    flat_centered = flat - flat.mean(dim=0)

    # Subsample for speed
    max_points = 2000
    if flat_centered.shape[0] > max_points:
        random.seed(seed)
        idx = sorted(random.sample(range(flat_centered.shape[0]), max_points))
        flat_centered = flat_centered[idx]
        ch_labels = np.array([i % n_channels for i in idx])
    else:
        ch_labels = np.tile(np.arange(n_channels), n_sites)

    _, s, v = torch.svd_lowrank(flat_centered, q=10)
    proj = (flat_centered @ v[:, :2]).numpy()

    colors = ["#c44e52", "#dd8452", "#55a868", "#ccb974", "#4c72b0"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_channels):
        mask = ch_labels == i
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.4, color=colors[i], label=CH_SHORT[i])

    ax.legend(markerscale=3, fontsize=9)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of CLS Features (colored by channel)")
    fig.tight_layout()

    out = output_dir / "pca_channels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_pca_variance(metrics: dict, output_dir: Path) -> Path:
    """Save PCA cumulative explained variance curve."""
    components = metrics["pca"]["components"]
    pcs = [c["pc"] for c in components]
    cumul = [c["cumulative"] for c in components]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pcs, cumul, "o-", markersize=4, color="#4c72b0")
    ax.axhline(90, color="red", linestyle="--", linewidth=1, label="90%")
    n_90 = metrics["pca"]["n_components_90pct"]
    if n_90:
        ax.axvline(n_90, color="gray", linestyle=":", linewidth=1)
        ax.annotate(
            f"{n_90} PCs",
            xy=(n_90, 90),
            xytext=(n_90 + 3, 82),
            arrowprops={"arrowstyle": "->", "color": "gray"},
            fontsize=9,
        )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Variance (%)")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    ax.set_ylim(0, 102)
    fig.tight_layout()

    out = output_dir / "pca_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_variance_decomp(metrics: dict, output_dir: Path) -> Path:
    """Save variance decomposition bar chart."""
    vd = metrics["variance_decomposition"]
    labels = ["Between\nChannel", "Between\nSite", "Residual\n(interaction)"]
    values = [vd["between_channel_pct"], vd["between_site_pct"], vd["residual_pct"]]
    colors = ["#55a868", "#4c72b0", "#ccb974"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
        )
    ax.set_ylabel("Variance (%)")
    ax.set_title("Feature Variance Decomposition")
    ax.set_ylim(0, max(values) + 10)
    fig.tight_layout()

    out = output_dir / "variance_decomp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# main


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DINOv3 feature extraction quality.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yml"))
    parser.add_argument("--plate", type=str, help="Diagnose a specific plate.")
    parser.add_argument(
        "--max-sites", type=int, default=500, help="Max sites to sample (default: 500)."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)["cpjump"]

    local = config.get("local", {})
    features_root = Path(local.get("features", "data/features"))

    if args.plate:
        plates = [args.plate]
    else:
        plates = [d.name for d in sorted(features_root.iterdir()) if d.is_dir()]

    if not plates:
        console.print("[red]No feature directories found.[/red]")
        return

    for plate in plates:
        feature_dir = features_root / plate
        if not feature_dir.exists():
            console.print(f"[red]Not found: {feature_dir}[/red]")
            continue

        out_dir = args.output_dir or Path("data/visualizations") / plate
        out_dir.mkdir(parents=True, exist_ok=True)

        n_files = len(list(feature_dir.glob("*.pt")))
        console.rule(f"[bold blue]Diagnosing {plate} ({n_files} sites)")

        # Load
        features, names = load_features(feature_dir, max_sites=args.max_sites)
        console.print(f"  Loaded {features.shape[0]} sites, shape: {tuple(features.shape)}")

        # Compute metrics
        console.print("  Computing metrics...")
        metrics = compute_metrics(features)

        # Save JSON report
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        console.print(f"  [dim]{metrics_path}[/dim]")

        # Generate plots
        console.print("  Generating plots...")
        for plot_fn in [
            plot_similarity_heatmap,
            plot_norm_distributions,
            plot_pca_channels,
            plot_pca_variance,
            plot_variance_decomp,
        ]:
            if plot_fn in (plot_norm_distributions, plot_pca_channels):
                path = plot_fn(features, out_dir)
            else:
                path = plot_fn(metrics, out_dir)
            console.print(f"  [dim]{path}[/dim]")

        # Print verdicts
        console.print()
        for v in metrics["verdicts"]:
            if v.startswith("PASS"):
                console.print(f"  [green]{v}[/green]")
            elif v.startswith("WARN"):
                console.print(f"  [yellow]{v}[/yellow]")
            else:
                console.print(f"  [red]{v}[/red]")

        # Key numbers
        console.print()
        off_diag_mean = metrics["intra_site_similarity"]["off_diagonal_mean"]
        console.print(f"  Inter-channel cosine (off-diag mean): {off_diag_mean}")
        console.print(
            f"  Cross-site channel gap:               {metrics['cross_site_similarity']['gap']}"
        )
        vd = metrics["variance_decomposition"]
        console.print(
            f"  Variance - channel: {vd['between_channel_pct']}% | "
            f"site: {vd['between_site_pct']}% | residual: {vd['residual_pct']}%"
        )
        pca_90 = metrics["pca"]["n_components_90pct"]
        console.print(f"  PCA 90% at {pca_90} components (of {metrics['feature_dim']})")

    console.print("\n[bold green]Done.")


if __name__ == "__main__":
    main()
