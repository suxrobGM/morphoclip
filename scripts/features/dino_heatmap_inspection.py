"""Render a DINO attention heatmap next to a microscopy image."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from morphoclip.data.feature_extractor import DEFAULT_MODEL  # noqa: E402
from morphoclip.data.image_loader import parse_filename  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("output/dino_heatmap_inspection")
DEFAULT_RAW_ROOTS = (Path("data/raw/images"), Path("data/raw"))
DEFAULT_COMPRESSED_ROOTS = (Path("data/raw_compressed"),)
SUPPORTED_MODES = ("nearest", "bilinear", "bicubic")
SUPPORTED_HEATMAP_METHODS = ("last_cls", "last4_cls", "rollout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a CPJUMP microscopy image, run DINO attention rollout, and "
            "save a side-by-side heatmap panel."
        )
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional path to a specific image file.",
    )
    parser.add_argument(
        "--source",
        choices=("auto", "raw", "raw_compressed"),
        default="raw_compressed",
        help="Input tree to inspect when --image is not provided.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Optional path to a specific plate Images/ directory.",
    )
    parser.add_argument(
        "--plate",
        type=str,
        default=None,
        help="Optional plate substring filter, e.g. BR00116991.",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=5,
        help="Microscopy channel to inspect.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=384,
        help="Target square size used before feeding the image to DINO.",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="bilinear",
        help="Interpolation mode used for resizing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Alpha value for the overlay panel.",
    )
    parser.add_argument(
        "--heatmap-method",
        choices=SUPPORTED_HEATMAP_METHODS,
        default="last4_cls",
        help="How to convert DINO attentions into a spatial heatmap.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the saved heatmap panel.",
    )
    return parser.parse_args()


def has_valid_images(image_dir: Path, channel: int) -> bool:
    try:
        for path in image_dir.iterdir():
            if not path.is_file():
                continue
            parsed = parse_filename(path.name)
            if parsed is None:
                continue
            _, parsed_channel = parsed
            if parsed_channel == channel:
                return True
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}") from exc
    return False


def iter_candidate_image_dirs(root: Path, plate: str | None) -> Iterable[Path]:
    if not root.exists():
        return
    for image_dir in root.rglob("Images"):
        if not image_dir.is_dir():
            continue
        if plate and plate not in image_dir.as_posix():
            continue
        yield image_dir


def resolve_input_dir(
    source: str,
    input_dir: Path | None,
    plate: str | None,
    channel: int,
) -> tuple[str, Path]:
    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        inferred_source = "raw_compressed" if "raw_compressed" in input_dir.as_posix() else "raw"
        if not has_valid_images(input_dir, channel):
            raise FileNotFoundError(f"No channel {channel} images found in {input_dir}")
        return inferred_source, input_dir

    search_order: list[tuple[str, tuple[Path, ...]]] = []
    if source == "auto":
        search_order = [
            ("raw_compressed", DEFAULT_COMPRESSED_ROOTS),
            ("raw", DEFAULT_RAW_ROOTS),
        ]
    elif source == "raw":
        search_order = [("raw", DEFAULT_RAW_ROOTS)]
    else:
        search_order = [("raw_compressed", DEFAULT_COMPRESSED_ROOTS)]

    for resolved_source, roots in search_order:
        for root in roots:
            for image_dir in iter_candidate_image_dirs(root, plate):
                if has_valid_images(image_dir, channel):
                    return resolved_source, image_dir

    if plate:
        raise FileNotFoundError(
            f"No Images/ directory with channel {channel} files found for plate '{plate}'."
        )
    raise FileNotFoundError(
        f"No Images/ directory with channel {channel} files found under the configured roots."
    )


def pick_image_path(
    image: Path | None,
    source: str,
    input_dir: Path | None,
    plate: str | None,
    channel: int,
) -> tuple[str, Path]:
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        inferred_source = "raw_compressed" if "raw_compressed" in image.as_posix() else "raw"
        return inferred_source, image

    resolved_source, image_dir = resolve_input_dir(source, input_dir, plate, channel)
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        _, parsed_channel = parsed
        if parsed_channel == channel:
            return resolved_source, path

    raise FileNotFoundError(f"No channel {channel} image found in {image_dir}")


def load_grayscale_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got shape {arr.shape}")
    return arr


def to_unit_float(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        raise ValueError("Cannot normalize an empty image.")

    arr_float = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        if max_value <= 0:
            raise ValueError(f"Unsupported integer dtype for normalization: {arr.dtype}")
        return arr_float / max_value

    arr_min = float(arr_float.min())
    arr_max = float(arr_float.max())
    if arr_max <= arr_min:
        return np.zeros_like(arr_float, dtype=np.float32)
    return (arr_float - arr_min) / (arr_max - arr_min)


def resize_float_image(arr: np.ndarray, height: int, width: int, mode: str) -> np.ndarray:
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    kwargs: dict[str, object] = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, size=(height, width), **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()


def build_model_input(
    processor: AutoImageProcessor,
    image_float: np.ndarray,
    target_size: int,
    mode: str,
) -> tuple[torch.Tensor, np.ndarray]:
    resized = resize_float_image(image_float, target_size, target_size, mode)
    rgb = np.repeat(resized[None, :, :], 3, axis=0)
    tensor = torch.from_numpy(rgb).unsqueeze(0)

    mean = torch.tensor(processor.image_mean, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=tensor.dtype).view(1, 3, 1, 1)
    normalized = (tensor - mean) / std
    return normalized, resized


def load_dino_with_attentions(
    model_name: str,
    device: str,
) -> tuple[AutoImageProcessor, torch.nn.Module]:
    try:
        processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        try:
            model = AutoModel.from_pretrained(
                model_name,
                attn_implementation="eager",
                local_files_only=True,
            )
        except TypeError:
            model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        processor = AutoImageProcessor.from_pretrained(model_name)
        try:
            model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        except TypeError:
            model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return processor, model


def compute_attention_rollout(
    attentions: tuple[torch.Tensor, ...],
    num_register_tokens: int = 0,
) -> np.ndarray:
    if not attentions:
        raise ValueError("Model did not return attention tensors.")

    num_tokens = attentions[0].shape[-1]
    rollout = torch.eye(num_tokens, device=attentions[0].device)

    for layer_attention in attentions:
        layer_attention = layer_attention.mean(dim=1).squeeze(0)
        layer_attention = layer_attention + torch.eye(num_tokens, device=layer_attention.device)
        layer_attention = layer_attention / layer_attention.sum(dim=-1, keepdim=True)
        rollout = layer_attention @ rollout

    patch_start = 1 + max(num_register_tokens, 0)
    cls_attention = rollout[0, patch_start:]
    num_patches = cls_attention.numel()
    grid_size = int(round(num_patches**0.5))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch tokens into a square heatmap.")

    heatmap = cls_attention.reshape(grid_size, grid_size).detach().cpu().numpy()
    heatmap -= heatmap.min()
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32, copy=False)
    heatmap -= heatmap.min()
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap /= max_value
    return heatmap


def compute_cls_attention_map(
    attentions: tuple[torch.Tensor, ...],
    num_register_tokens: int,
    layers: int,
) -> np.ndarray:
    if not attentions:
        raise ValueError("Model did not return attention tensors.")
    if layers <= 0:
        raise ValueError("layers must be positive.")

    patch_start = 1 + max(num_register_tokens, 0)
    selected_layers = attentions[-layers:]
    cls_maps = [
        layer_attention.mean(dim=1)[0, 0, patch_start:] for layer_attention in selected_layers
    ]
    cls_attention = torch.stack(cls_maps, dim=0).mean(dim=0)

    num_patches = cls_attention.numel()
    grid_size = int(round(num_patches**0.5))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patch tokens into a square heatmap.")

    return normalize_heatmap(cls_attention.reshape(grid_size, grid_size).detach().cpu().numpy())


def build_attention_heatmap(
    attentions: tuple[torch.Tensor, ...],
    method: str,
    num_register_tokens: int,
) -> np.ndarray:
    if method == "rollout":
        return compute_attention_rollout(attentions, num_register_tokens=num_register_tokens)
    if method == "last_cls":
        return compute_cls_attention_map(
            attentions, num_register_tokens=num_register_tokens, layers=1
        )
    if method == "last4_cls":
        return compute_cls_attention_map(
            attentions, num_register_tokens=num_register_tokens, layers=4
        )
    raise ValueError(f"Unsupported heatmap method: {method}")


def render_panel(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay_alpha: float,
    image_path: Path,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("Original")
    axes[0].axis("off")

    heat = axes[1].imshow(heatmap, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1].set_title("DINO Heatmap")
    axes[1].axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].imshow(heatmap, cmap="inferno", vmin=0.0, vmax=1.0, alpha=overlay_alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(image_path.name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_method_comparison_panel(
    original: np.ndarray,
    heatmaps: dict[str, np.ndarray],
    overlay_alpha: float,
    image_path: Path,
    output_path: Path,
) -> None:
    ordered_methods = [method for method in SUPPORTED_HEATMAP_METHODS if method in heatmaps]
    fig, axes = plt.subplots(
        2, len(ordered_methods) + 1, figsize=(5 * (len(ordered_methods) + 1), 9)
    )

    axes[0, 0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    for column, method in enumerate(ordered_methods, start=1):
        heatmap = heatmaps[method]
        heat = axes[0, column].imshow(heatmap, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(f"{method} heatmap")
        axes[0, column].axis("off")
        fig.colorbar(heat, ax=axes[0, column], fraction=0.046, pad=0.04)

        axes[1, column].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, column].imshow(heatmap, cmap="inferno", vmin=0.0, vmax=1.0, alpha=overlay_alpha)
        axes[1, column].set_title(f"{method} overlay")
        axes[1, column].axis("off")

    fig.suptitle(f"{image_path.name} attention comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()

    resolved_source, image_path = pick_image_path(
        image=args.image,
        source=args.source,
        input_dir=args.input_dir,
        plate=args.plate,
        channel=args.channel,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_raw = load_grayscale_array(image_path)
    image_float = to_unit_float(image_raw)

    processor, model = load_dino_with_attentions(args.model_name, args.device)

    inputs, resized = build_model_input(
        processor=processor,
        image_float=image_float,
        target_size=args.target_size,
        mode=args.mode,
    )
    outputs = model(
        pixel_values=inputs.to(args.device),
        output_attentions=True,
    )
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    heatmaps_by_method: dict[str, np.ndarray] = {}
    for method in SUPPORTED_HEATMAP_METHODS:
        heatmap_small = build_attention_heatmap(
            outputs.attentions,
            method=method,
            num_register_tokens=num_register_tokens,
        )
        heatmaps_by_method[method] = normalize_heatmap(
            resize_float_image(
                heatmap_small.astype(np.float32),
                image_float.shape[0],
                image_float.shape[1],
                args.mode,
            )
        )

    heatmap = heatmaps_by_method[args.heatmap_method]

    stem = image_path.stem
    panel_path = args.output_dir / f"{stem}_dino_heatmap.png"
    comparison_path = args.output_dir / f"{stem}_dino_heatmap_comparison.png"
    resized_path = args.output_dir / f"{stem}_resized_input.png"
    heatmap_path = args.output_dir / f"{stem}_heatmap.npy"

    Image.fromarray(np.rint(np.clip(resized, 0.0, 1.0) * 255.0).astype(np.uint8)).save(resized_path)
    np.save(heatmap_path, heatmap.astype(np.float32))
    render_panel(
        original=image_float,
        heatmap=heatmap,
        overlay_alpha=args.overlay_alpha,
        image_path=image_path,
        output_path=panel_path,
    )
    render_method_comparison_panel(
        original=image_float,
        heatmaps=heatmaps_by_method,
        overlay_alpha=args.overlay_alpha,
        image_path=image_path,
        output_path=comparison_path,
    )

    print(f"source={resolved_source}")
    print(f"image={image_path}")
    print(f"heatmap_method={args.heatmap_method}")
    print(f"panel={panel_path}")
    print(f"comparison_panel={comparison_path}")
    print(f"resized_input={resized_path}")
    print(f"heatmap={heatmap_path}")


if __name__ == "__main__":
    main()
