"""CPJUMP1 image loading and preprocessing.

Handles parsing microscopy filenames across raw and compressed formats,
channel loading, numeric normalization, and resizing for DINOv3 input.

CPJUMP1 Image Naming Convention:
    ``r{row}c{col}f{field}p{plane}-ch{channel}sk1fk1fl1.<ext>``

Channel mapping (Cell Painting):
    ch1: Mitochondria (MitoTracker / Alexa 647)
    ch2: Actin (Phalloidin / Alexa 568)
    ch3: Golgi/Plasma Membrane (WGA / Alexa 488 long)
    ch4: Endoplasmic Reticulum (Concanavalin A / Alexa 488)
    ch5: DNA/Nucleus (Hoechst 33342)
    ch6-ch8: Brightfield z-planes (not used)
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Fluorescence channels used for Cell Painting (skip ch6-ch8 brightfield)
FLUORESCENCE_CHANNELS: tuple[int, ...] = (1, 2, 3, 4, 5)

CHANNEL_NAMES: dict[int, str] = {
    1: "Mitochondria",
    2: "Actin",
    3: "Golgi/Plasma Membrane",
    4: "Endoplasmic Reticulum",
    5: "DNA/Nucleus",
}

DINO_INPUT_SIZE: int = 384

# ImageNet normalization constants used by DINOv3
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_FILENAME_PATTERN = re.compile(
    r"r(?P<row>\d{2})c(?P<col>\d{2})f(?P<field>\d{2})p(?P<plane>\d{2})"
    r"-ch(?P<channel>\d+)sk1fk1fl1\.(?:tiff|tif|jpg|jpeg|jp2|webp|png)$",
    re.IGNORECASE,
)

# Pattern for saved feature/tensor .pt files: r01c01f01.pt
FEATURE_PATTERN = re.compile(r"r(?P<row>\d{2})c(?P<col>\d{2})f(?P<field>\d{2})\.pt")

_UINT16_MAX = 65535.0


@dataclass(frozen=True, slots=True)
class ImageKey:
    """Unique identifier for a single imaging site (well + field).

    Each site has multiple channels; this key groups them.
    """

    row: int
    col: int
    field: int
    plate: str = ""

    @property
    def well(self) -> str:
        """Well position in letter-number format (e.g. ``"A01"``)."""
        return f"{chr(64 + self.row)}{self.col:02d}"

    def __str__(self) -> str:
        return f"r{self.row:02d}c{self.col:02d}f{self.field:02d}"


def parse_filename(filename: str) -> tuple[ImageKey, int] | None:
    """Parse a CPJUMP1 microscopy filename into its components.

    Args:
        filename: Microscopy filename like
            ``"r01c01f01p01-ch1sk1fk1fl1.tiff"`` or
            ``"r01c01f01p01-ch1sk1fk1fl1.jpg"``.

    Returns:
        ``(ImageKey, channel_number)`` or ``None`` if the filename
        doesn't match the expected pattern.
    """
    match = _FILENAME_PATTERN.match(filename)
    if match is None:
        return None
    return (
        ImageKey(
            row=int(match["row"]),
            col=int(match["col"]),
            field=int(match["field"]),
        ),
        int(match["channel"]),
    )


def discover_sites(
    image_dir: Path,
    channels: tuple[int, ...] = FLUORESCENCE_CHANNELS,
) -> dict[ImageKey, dict[int, Path]]:
    """Discover all imaging sites in a plate directory.

    Scans the directory for microscopy image files, groups them by
    (row, col, field), and maps each site to its channel file paths.

    Args:
        image_dir: Path to the plate's ``Images/`` directory.
        channels: Channel numbers to require (default: fluorescence only).

    Returns:
        Dict mapping ``ImageKey -> {channel_num: Path}``.
        Only sites with all requested channels are included.
    """
    channel_set = set(channels)
    sites: dict[ImageKey, dict[int, Path]] = {}

    for path in sorted(image_dir.iterdir()):
        parsed = parse_filename(path.name)
        if parsed is None:
            continue
        key, ch = parsed
        if ch not in channel_set:
            continue
        sites.setdefault(key, {})[ch] = path

    complete = {key: ch_paths for key, ch_paths in sites.items() if len(ch_paths) == len(channels)}
    return complete


def load_single_channel(path: Path, normalize: bool = True) -> np.ndarray:
    """Load a single microscopy image as a float32 array.

    Args:
        path: Path to the image file.
        normalize: If ``True``, normalize integer-valued images to
            float32 ``[0, 1]`` based on their dtype range.

    Returns:
        2D array of shape ``(H, W)``, dtype ``float32``.
    """
    img = Image.open(path)
    raw = np.array(img)
    original_dtype = raw.dtype
    arr = raw
    if arr.ndim != 2:
        raise ValueError(f"Expected a single-channel image at {path}, got shape {arr.shape}")

    arr = arr.astype(np.float32)
    if normalize:
        if np.issubdtype(original_dtype, np.integer):
            arr /= float(np.iinfo(original_dtype).max)
    return arr


def load_site(
    channel_paths: dict[int, Path],
    channels: tuple[int, ...] = FLUORESCENCE_CHANNELS,
    normalize: bool = True,
) -> np.ndarray:
    """Load and stack all channels for a single site.

    Args:
        channel_paths: Mapping of ``channel_number -> file_path``.
        channels: Channel numbers to load, in order.
        normalize: Whether to normalize to ``[0, 1]``.

    Returns:
        3D array of shape ``(C, H, W)``, dtype ``float32``.

    Raises:
        FileNotFoundError: If a requested channel file is missing.
    """
    arrays = []
    for ch in channels:
        if ch not in channel_paths:
            raise FileNotFoundError(f"Channel {ch} not found in {channel_paths}")
        arrays.append(load_single_channel(channel_paths[ch], normalize=normalize))
    return np.stack(arrays, axis=0)


def load_site_as_tensor(
    channel_paths: dict[int, Path],
    channels: tuple[int, ...] = FLUORESCENCE_CHANNELS,
    normalize: bool = True,
    resize: int | None = None,
) -> torch.Tensor:
    """Load a site as a PyTorch tensor, optionally resized.

    Args:
        channel_paths: Mapping of ``channel_number -> file_path``.
        channels: Channel numbers to load, in order.
        normalize: Whether to normalize to ``[0, 1]``.
        resize: If provided, resize spatial dims to ``(resize, resize)``
            using bilinear interpolation.

    Returns:
        Tensor of shape ``(C, H, W)`` or ``(C, resize, resize)``.
    """
    arr = load_site(channel_paths, channels=channels, normalize=normalize)
    tensor = torch.from_numpy(arr)

    if resize is not None:
        # F.interpolate expects (N, C, H, W)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return tensor


def prepare_channels_for_dino(
    site_tensor: torch.Tensor,
    apply_imagenet_norm: bool = False,
) -> torch.Tensor:
    """Prepare multi-channel site tensor for per-channel DINOv3 input.

    DINOv3 expects 3-channel (RGB) input. Each Cell Painting channel is
    replicated to 3 channels to form a pseudo-RGB image.

    Args:
        site_tensor: Tensor of shape ``(C, H, W)`` where ``C`` is the
            number of fluorescence channels (typically 5). Values should
            be in ``[0, 1]``.
        apply_imagenet_norm: If ``True``, apply ImageNet normalization
            after replication. The ``AutoImageProcessor`` usually handles
            this, so this is ``False`` by default.

    Returns:
        Tensor of shape ``(C, 3, H, W)`` — a batch of ``C`` pseudo-RGB
        images, one per fluorescence channel.
    """
    num_channels = site_tensor.shape[0]
    # (C, H, W) -> (C, 1, H, W) -> (C, 3, H, W)
    batch = site_tensor.unsqueeze(1).expand(num_channels, 3, -1, -1).clone()

    if apply_imagenet_norm:
        mean = torch.tensor(IMAGENET_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=batch.dtype).view(1, 3, 1, 1)
        batch = (batch - mean) / std

    return batch
