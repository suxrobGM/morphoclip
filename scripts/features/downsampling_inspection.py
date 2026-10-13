"""Downsample a demo set of CPJUMP images and render inspection panels."""

import csv
import random
import sys
from pathlib import Path
from typing import Annotated

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import typer
from PIL import Image
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from image_sources import (
    InterpMode,
    Source,
    load_grayscale_array,
    resize_float_image,
    resolve_input_dir,
    to_unit_float,
)

from morphoclip.data.image_loader import FLUORESCENCE_CHANNELS, parse_filename

DEFAULT_OUTPUT_DIR = Path("output/downsampling_inspection")

console = Console()


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


def main(
    source: Annotated[
        Source, typer.Option(help="Input tree to inspect. auto prefers raw_compressed, then raw.")
    ] = Source.auto,
    input_dir: Annotated[
        Path | None, typer.Option(help="Optional path to a specific plate Images/ directory.")
    ] = None,
    plate: Annotated[
        str | None, typer.Option(help="Optional plate substring filter, e.g. BR00116991.")
    ] = None,
    num_images: Annotated[int, typer.Option(help="How many demo images to sample.")] = 10,
    target_size: Annotated[int, typer.Option(help="Target square size for downsampling.")] = 384,
    channels: Annotated[
        list[int], typer.Option(help="Channels to sample from. Defaults to fluorescence 1-5.")
    ] = list(FLUORESCENCE_CHANNELS),  # noqa: B006 - Typer copies list defaults per call
    seed: Annotated[int, typer.Option(help="Random seed used when sampling demo images.")] = 56,
    mode: Annotated[
        InterpMode, typer.Option(help="Interpolation mode for downsampling and reconstruction.")
    ] = InterpMode.bilinear,
    output_dir: Annotated[Path, typer.Option(help="Root output directory.")] = DEFAULT_OUTPUT_DIR,
) -> None:
    """Sample microscopy images, downsample to a square size, and save comparison panels."""
    channel_set = set(channels)
    resolved_source, image_dir = resolve_input_dir(source, input_dir, plate, channel_set)
    all_images = discover_images(image_dir, channel_set)
    sampled_images = sample_demo_images(all_images, num_images, seed)

    if not sampled_images:
        console.print(f"[red]No matching images found in {image_dir}[/red]")
        raise typer.Exit(1)

    output_root = output_dir / resolved_source / image_dir.parent.name
    downsampled_dir = output_root / "downsampled"
    panels_dir = output_root / "panels"
    manifest_path = output_root / "manifest.csv"

    rows: list[dict[str, object]] = []
    for path in sampled_images:
        original = load_grayscale_array(path)
        original_unit = to_unit_float(original)
        downsampled_unit = resize_float_image(
            original_unit,
            height=target_size,
            width=target_size,
            mode=mode,
        )
        reconstructed_unit = resize_float_image(
            downsampled_unit,
            height=original.shape[0],
            width=original.shape[1],
            mode=mode,
        )

        downsampled = restore_dtype(downsampled_unit, original.dtype)
        reconstructed = restore_dtype(reconstructed_unit, original.dtype)

        suffix = output_suffix(original.dtype)
        downsampled_path = downsampled_dir / f"{path.stem}_{target_size}x{target_size}{suffix}"
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
            title=f"{resolved_source} | {path.name} | channel {channel} | mode={mode}",
        )

        rows.append(
            {
                "source": resolved_source,
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

    console.print(f"Source: {resolved_source}")
    console.print(f"Image dir: {image_dir}")
    console.print(f"Sampled images: {len(sampled_images)}")
    console.print(f"Downsampled outputs: {downsampled_dir}")
    console.print(f"Comparison panels: {panels_dir}")
    console.print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    typer.run(main)
