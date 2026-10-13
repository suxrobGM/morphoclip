"""Locate and load raw CPJUMP1 microscopy images for the inspection scripts."""

from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from morphoclip.data.image_loader import parse_filename

DEFAULT_RAW_ROOTS = (Path("data/raw/images"), Path("data/raw"))
DEFAULT_COMPRESSED_ROOTS = (Path("data/raw_compressed"),)


class Source(StrEnum):
    auto = "auto"
    raw = "raw"
    raw_compressed = "raw_compressed"


class InterpMode(StrEnum):
    nearest = "nearest"
    bilinear = "bilinear"
    bicubic = "bicubic"


def has_valid_images(image_dir: Path, channels: set[int]) -> bool:
    """Return whether ``image_dir`` holds at least one image of one of ``channels``."""
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        _, channel = parsed
        if channel in channels:
            return True
    return False


def iter_candidate_image_dirs(root: Path, plate: str | None) -> Iterable[Path]:
    """Yield every ``Images/`` directory under ``root``, optionally filtered by plate."""
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
    """Find an ``Images/`` directory with images of ``channels``.

    Returns:
        ``(source, image_dir)``, where ``source`` is ``"raw"`` or ``"raw_compressed"``.

    Raises:
        FileNotFoundError: If no matching directory exists.
    """
    if input_dir is not None:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        inferred_source = "raw_compressed" if "raw_compressed" in input_dir.as_posix() else "raw"
        if not has_valid_images(input_dir, channels):
            raise FileNotFoundError(f"No channel {sorted(channels)} images found in {input_dir}")
        return inferred_source, input_dir

    search_order: list[tuple[str, tuple[Path, ...]]]
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
            f"No Images/ directory with channel {sorted(channels)} files found for plate '{plate}'."
        )
    raise FileNotFoundError(
        f"No Images/ directory with channel {sorted(channels)} files found under the "
        "configured roots."
    )


def load_grayscale_array(path: Path) -> np.ndarray:
    """Load a single-channel image as a 2D array."""
    arr = np.asarray(Image.open(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image at {path}, got shape {arr.shape}")
    return arr


def to_unit_float(arr: np.ndarray) -> np.ndarray:
    """Scale an image to ``[0, 1]`` float32 (by dtype range for integers, min-max otherwise)."""
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
    """Resize a 2D float image with ``torch.nn.functional.interpolate``."""
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    kwargs: dict[str, object] = {"mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = F.interpolate(tensor, size=(height, width), **kwargs)
    return resized.squeeze(0).squeeze(0).numpy()
