"""Downsample a demo set of CPJUMP images and render inspection panels."""

from __future__ import annotations

import argparse
import csv
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morphoclip.data.image_loader import FLUORESCENCE_CHANNELS, parse_filename  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("output/downsampling_inspection")
DEFAULT_RAW_ROOTS = (Path("data/raw/images"), Path("data/raw"))
DEFAULT_COMPRESSED_ROOTS = (Path("data/raw_compressed"),)
SUPPORTED_MODES = ("nearest", "bilinear", "bicubic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a small set of microscopy images from raw or raw_compressed, "
            "downsample them to 384x384, and save comparison panels."
        )
    )
    parser.add_argument(
        "--source",
        choices=("auto", "raw", "raw_compressed"),
        default="auto",
        help="Input tree to inspect. auto prefers raw_compressed, then raw.",
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
        "--num-images",
        type=int,
        default=10,
        help="How many demo images to sample.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=384,
        help="Target square size for downsampling.",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=list(FLUORESCENCE_CHANNELS),
        help="Channels to sample from. Defaults to fluorescence channels 1-5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=56,
        help="Random seed used when sampling demo images.",
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        default="bilinear",
        help="Interpolation mode used for downsampling and reconstruction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory.",
    )
    return parser.parse_args()


def has_valid_images(image_dir: Path, channels: set[int]) -> bool:
    try:
        for path in image_dir.iterdir():
            if not path.is_file():
                continue
            parsed = parse_filename(path.name)
            if parsed is None:
                continue
            _, channel = parsed
            if channel in channels:
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
    channels: set[int],
) -> tuple[str, Path]:
    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        inferred_source = "raw_compressed" if "raw_compressed" in input_dir.as_posix() else "raw"
        if not has_valid_images(input_dir, channels):
            raise FileNotFoundError(f"No matching images found in {input_dir}")
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
                if has_valid_images(image_dir, channels):
                    return resolved_source, image_dir

    if plate:
        raise FileNotFoundError(
            f"No Images/ directory with matching files found for plate '{plate}'."
        )
    raise FileNotFoundError(
        "No Images/ directory with matching files found under the configured roots."
    )


def discover_images(image_dir: Path, channels: set[int]) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        _, channel = parsed
        if channel not in channels:
            continue
        paths.append(path)
    return paths


def sample_demo_images(paths: list[Path], num_images: int, seed: int) -> list[Path]:
    if num_images <= 0:
        raise ValueError("--num-images must be positive.")
    if not paths:
        return []

    rng = random.Random(seed)
    by_channel: dict[int, list[Path]] = {}
    for path in paths:
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        _, channel = parsed
        by_channel.setdefault(channel, []).append(path)

    for bucket in by_channel.values():
        rng.shuffle(bucket)

    ordered_channels = sorted(by_channel)
    sampled: list[Path] = []
    target = min(num_images, len(paths))

    while len(sampled) < target:
        made_progress = False
        for channel in ordered_channels:
            bucket = by_channel[channel]
            if not bucket:
                continue
            sampled.append(bucket.pop())
            made_progress = True
            if len(sampled) == target:
                break
        if not made_progress:
            break

    return sampled


def load_grayscale_array(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got shape {arr.shape}")
    return arr


def to_unit_float(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        raise ValueError("Cannot resize an empty image.")

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


def restore_dtype(arr: np.ndarray, dtype: np.dtype) -> np.ndarray:
    clipped = np.clip(arr, 0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        max_value = float(np.iinfo(dtype).max)
        return np.rint(clipped * max_value).astype(dtype)
    return clipped.astype(dtype)


def output_suffix(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.uint16):
        return ".tiff"
    if np.issubdtype(dtype, np.integer):
        return ".png"
    return ".tiff"


def to_display_uint8(arr: np.ndarray) -> np.ndarray:
    arr_float = arr.astype(np.float32)
    lo = float(np.percentile(arr_float, 1.0))
    hi = float(np.percentile(arr_float, 99.0))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr_float - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(scaled * 255.0).astype(np.uint8)


def render_panel(
    original: np.ndarray,
    downsampled: np.ndarray,
    reconstructed: np.ndarray,
    panel_path: Path,
    title: str,
) -> None:
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
    diff_disp = to_display_uint8(diff)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(title)

    axes[0].imshow(to_display_uint8(original), cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f"Original\n{original.shape[1]}x{original.shape[0]}")
    axes[0].axis("off")

    axes[1].imshow(to_display_uint8(downsampled), cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Downsampled\n{downsampled.shape[1]}x{downsampled.shape[0]}")
    axes[1].axis("off")

    axes[2].imshow(to_display_uint8(reconstructed), cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Reconstructed\nback to original size")
    axes[2].axis("off")

    axes[3].imshow(diff_disp, cmap="magma", vmin=0, vmax=255)
    axes[3].set_title("Abs diff")
    axes[3].axis("off")

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(panel_path, dpi=150)
    plt.close(fig)


def write_manifest(rows: list[dict[str, object]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "image_dir",
        "filename",
        "image_id",
        "channel",
        "original_shape",
        "downsampled_path",
        "panel_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    channels = set(args.channels)
    source, image_dir = resolve_input_dir(args.source, args.input_dir, args.plate, channels)
    all_images = discover_images(image_dir, channels)
    sampled_images = sample_demo_images(all_images, args.num_images, args.seed)

    if not sampled_images:
        raise FileNotFoundError(f"No matching images found in {image_dir}")

    output_root = args.output_dir / source / image_dir.parent.name
    downsampled_dir = output_root / "downsampled"
    panels_dir = output_root / "panels"
    manifest_path = output_root / "manifest.csv"

    rows: list[dict[str, object]] = []
    for path in sampled_images:
        original = load_grayscale_array(path)
        original_unit = to_unit_float(original)
        downsampled_unit = resize_float_image(
            original_unit,
            height=args.target_size,
            width=args.target_size,
            mode=args.mode,
        )
        reconstructed_unit = resize_float_image(
            downsampled_unit,
            height=original.shape[0],
            width=original.shape[1],
            mode=args.mode,
        )

        downsampled = restore_dtype(downsampled_unit, original.dtype)
        reconstructed = restore_dtype(reconstructed_unit, original.dtype)

        suffix = output_suffix(original.dtype)
        downsampled_path = downsampled_dir / f"{path.stem}_384x384{suffix}"
        downsampled_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(downsampled).save(downsampled_path)

        parsed = parse_filename(path.name)
        if parsed is None:
            raise ValueError(f"Unexpected CPJUMP filename: {path.name}")
        image_key, channel = parsed

        panel_path = panels_dir / f"{path.stem}.png"
        render_panel(
            original=original,
            downsampled=downsampled,
            reconstructed=reconstructed,
            panel_path=panel_path,
            title=f"{source} | {path.name} | channel {channel} | mode={args.mode}",
        )

        rows.append(
            {
                "source": source,
                "image_dir": str(image_dir),
                "filename": path.name,
                "image_id": f"{image_key}-ch{channel}",
                "channel": channel,
                "original_shape": f"{original.shape[0]}x{original.shape[1]}",
                "downsampled_path": str(downsampled_path),
                "panel_path": str(panel_path),
            }
        )

    write_manifest(rows, manifest_path)

    print(f"Source: {source}")
    print(f"Image dir: {image_dir}")
    print(f"Sampled images: {len(sampled_images)}")
    print(f"Downsampled outputs: {downsampled_dir}")
    print(f"Comparison panels: {panels_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
