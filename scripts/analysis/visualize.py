"""Visualization utilities for Cell Painting image data.

Generates comparison images of original channels vs CLS features
for visual QC during feature extraction.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from morphoclip.data.image_loader import CHANNEL_NAMES, FLUORESCENCE_CHANNELS, ImageKey

# Pseudocolor LUTs per channel (approximate Cell Painting display colors)
CHANNEL_CMAPS: dict[int, str] = {
    1: "RdPu",  # Mitochondria (far-red / magenta-like)
    2: "Reds",  # Actin (red/orange)
    3: "Greens",  # Golgi/PM (green)
    4: "YlGn",  # ER (green/yellow)
    5: "Blues",  # DNA (blue)
}


def save_site_comparison(
    site_tensor: torch.Tensor,
    key: ImageKey,
    output_dir: Path,
    *,
    cls_features: torch.Tensor | None = None,
    channels: tuple[int, ...] = FLUORESCENCE_CHANNELS,
    contrast_percentile: float = 99.5,
    dpi: int = 120,
) -> Path:
    """Save a comparison figure: original channels vs CLS feature heatmaps.

    Produces a multi-row figure:

    - **Row 1** — Original fluorescence channels (pseudocolor, contrast-stretched).
    - **Row 2** — CLS token feature maps. Each channel's 1024-d CLS vector
      is reshaped to a 32x32 grid and displayed as a heatmap, giving a
      visual fingerprint of what DINOv3 extracted.

    If ``cls_features`` is ``None``, only row 1 is shown.

    Args:
        site_tensor: Tensor of shape ``(C, H, W)`` with values in ``[0, 1]``.
        key: ``ImageKey`` identifying the site.
        output_dir: Directory to save the PNG.
        cls_features: Optional CLS tokens of shape ``(C, D)`` where D is
            the model's hidden dimension (e.g. 1024 for ViT-L).
        channels: Channel numbers corresponding to the tensor's first dim.
        contrast_percentile: Upper percentile for contrast stretching.
        dpi: Output image DPI.

    Returns:
        Path to the saved PNG file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_channels = site_tensor.shape[0]
    arr = site_tensor.numpy() if isinstance(site_tensor, torch.Tensor) else site_tensor

    n_rows = 2 if cls_features is not None else 1
    fig, axes = plt.subplots(
        n_rows,
        n_channels,
        figsize=(3.5 * n_channels, 3.5 * n_rows),
        squeeze=False,
    )

    for i, ch in enumerate(channels):
        ax = axes[0, i]
        img = arr[i]
        vmin = 0.0
        vmax = float(np.percentile(img, contrast_percentile))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        cmap = CHANNEL_CMAPS.get(ch, "gray")
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(CHANNEL_NAMES.get(ch, f"ch{ch}"), fontsize=10)
        ax.axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=11, rotation=0, labelpad=60, va="center")

    if cls_features is not None:
        feat = cls_features.numpy() if isinstance(cls_features, torch.Tensor) else cls_features
        dim = feat.shape[1]
        side = int(math.isqrt(dim))
        if side * side < dim:
            side += 1
        padded_dim = side * side

        for i, _ch in enumerate(channels):
            ax = axes[1, i]
            vec = feat[i]
            if len(vec) < padded_dim:
                vec = np.pad(vec, (0, padded_dim - len(vec)))
            grid = vec[:padded_dim].reshape(side, side)
            ax.imshow(grid, cmap="viridis", aspect="equal")
            ax.set_title(f"CLS ({dim}-d)", fontsize=9)
            ax.axis("off")
        axes[1, 0].set_ylabel("CLS Token", fontsize=11, rotation=0, labelpad=60, va="center")

    fig.suptitle(f"{key}  (well {key.well}) - original vs CLS", fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = output_dir / f"{key}_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
