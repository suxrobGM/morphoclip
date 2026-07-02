"""Vision backbone feature extraction for CPJUMP1 Cell Painting images.

Extracts per-channel CLS tokens from 5 fluorescence channels per site,
saving ``(5, D)`` feature tensors as ``.pt`` files.

Default model: DINOv3 ViT-L/16 (300M params, 1024-dim CLS token).
Can also be used with DINOv2 and other Hugging Face vision backbones that
accept ``pixel_values`` and return either ``pooler_output`` or
``last_hidden_state`` with a CLS token.
"""

import logging
from pathlib import Path

import torch
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from transformers import AutoImageProcessor, AutoModel

from morphoclip.data.image_loader import (
    FLUORESCENCE_CHANNELS,
    ImageKey,
    discover_sites,
    load_site_as_tensor,
    prepare_channels_for_dino,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def infer_feature_width(model: torch.nn.Module) -> int:
    """Infer the CLS/pooler feature width produced by a HF vision backbone."""
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("Model does not expose a config; cannot infer feature width")

    for attr in ("hidden_size", "projection_dim", "embed_dim"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        for attr in ("hidden_size", "projection_dim", "embed_dim"):
            value = getattr(vision_config, attr, None)
            if isinstance(value, int) and value > 0:
                return value

    raise ValueError(
        f"Could not infer feature width from model config type {type(config).__name__}"
    )


def load_dinov3(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
) -> tuple[torch.nn.Module, AutoImageProcessor]:
    """Load a frozen Hugging Face vision model and its image processor.

    Args:
        model_name: HuggingFace model ID.
        device: Target device string.

    Returns:
        ``(model, processor)`` tuple. Model is in eval mode with
        all parameters frozen.
    """
    logger.info("Loading vision model: %s", model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Model loaded on %s", device)
    return model, processor


def _processor_input_size(processor: AutoImageProcessor) -> int:
    """Resolve the model's expected square input size from a HF image processor."""
    for attr in ("crop_size", "size"):
        value = getattr(processor, attr, None)
        if isinstance(value, dict):
            for key in ("height", "shortest_edge"):
                if key in value:
                    return int(value[key])
        if isinstance(value, int):
            return int(value)

    raise ValueError("Could not infer input size from processor.crop_size or processor.size")


@torch.no_grad()
def _extract_cls_tokens(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Extract CLS-like tokens ``(B, D)`` from preprocessed images ``(B, 3, H, W)``."""
    images = images.to(device)
    outputs = model(pixel_values=images)
    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output.cpu()

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise ValueError("Model output does not contain pooler_output or last_hidden_state")

    return last_hidden_state[:, 0].cpu()


def _preprocess_batch(
    processor: AutoImageProcessor,
    site_tensors: list[torch.Tensor],
) -> torch.Tensor:
    """Apply ImageNet normalization to pseudo-RGB site tensors.

    Args:
        processor: DINOv3 ``AutoImageProcessor`` (provides mean/std).
        site_tensors: List of tensors, each shape ``(5, 3, H, W)``.

    Returns:
        Normalized tensor of shape ``(N*5, 3, H, W)``.
    """
    all_images = torch.cat(site_tensors, dim=0)  # (N*5, 3, H, W)

    # Apply normalization manually — processor expects PIL images but we have tensors
    mean = torch.tensor(processor.image_mean, dtype=all_images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=all_images.dtype).view(1, 3, 1, 1)
    all_images = (all_images - mean) / std

    return all_images


def feature_filename(key: ImageKey) -> str:
    """Generate filename for a feature ``.pt`` file.

    Format matches :data:`~morphoclip.data.image_loader.FEATURE_PATTERN`.
    """
    return f"r{key.row:02d}c{key.col:02d}f{key.field:02d}.pt"


@torch.no_grad()
def extract_plate_features_with_model(
    image_dir: Path,
    output_dir: Path,
    model: torch.nn.Module,
    processor: AutoImageProcessor,
    *,
    device: str = "auto",
    batch_size: int = 32,
    save_tensors: bool = False,
    tensor_output_dir: Path | None = None,
    tensor_size: int | None = None,
) -> dict[ImageKey, Path]:
    """Extract DINOv3 features using a pre-loaded model.

    Like :func:`extract_plate_features` but accepts a pre-loaded model,
    avoiding redundant model loads when processing multiple plates.

    Args:
        image_dir: Path to the plate's ``Images/`` directory.
        output_dir: Directory to save feature ``.pt`` files.
        model: Pre-loaded, frozen DINOv3 model.
        processor: Corresponding ``AutoImageProcessor``.
        device: Torch device string.
        batch_size: Number of sites per GPU batch.
        save_tensors: Also save resized image tensors.
        tensor_output_dir: Directory for tensors (default: sibling ``tensors/``).
        tensor_size: Spatial size for resized images. If ``None``, use the
            processor's native crop/resize size.

    Returns:
        Dict mapping ``ImageKey -> Path`` to saved feature files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if tensor_size is None:
        tensor_size = _processor_input_size(processor)
    if save_tensors:
        if tensor_output_dir is None:
            tensor_output_dir = output_dir.parent.parent / "tensors" / output_dir.name
        tensor_output_dir.mkdir(parents=True, exist_ok=True)

    sites = discover_sites(image_dir, channels=FLUORESCENCE_CHANNELS)
    logger.info("Found %d complete sites in %s", len(sites), image_dir)

    if not sites:
        logger.warning("No complete sites found in %s", image_dir)
        return {}

    site_keys = sorted(sites.keys(), key=str)
    saved: dict[ImageKey, Path] = {}
    num_channels = len(FLUORESCENCE_CHANNELS)

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Extracting features", total=len(site_keys))

        for batch_start in range(0, len(site_keys), batch_size):
            batch_keys = site_keys[batch_start : batch_start + batch_size]
            batch_site_tensors: list[torch.Tensor] = []

            for key in batch_keys:
                site_tensor = load_site_as_tensor(sites[key], resize=tensor_size)  # (5, H, W)

                if save_tensors and tensor_output_dir is not None:
                    tensor_path = tensor_output_dir / feature_filename(key)
                    torch.save(site_tensor, tensor_path)

                dino_input = prepare_channels_for_dino(site_tensor)  # (5, 3, H, W)
                batch_site_tensors.append(dino_input)

            preprocessed = _preprocess_batch(processor, batch_site_tensors)

            all_cls: list[torch.Tensor] = []
            sub_batch_size = batch_size * num_channels
            for i in range(0, preprocessed.shape[0], sub_batch_size):
                sub = preprocessed[i : i + sub_batch_size]
                cls = _extract_cls_tokens(model, sub, device)
                all_cls.append(cls)
            all_cls_cat = torch.cat(all_cls, dim=0)  # (N*5, D)

            all_cls_reshaped = all_cls_cat.view(len(batch_keys), num_channels, -1)  # (N, 5, D)

            for i, key in enumerate(batch_keys):
                feat_path = output_dir / feature_filename(key)
                torch.save(all_cls_reshaped[i], feat_path)  # (5, D)
                saved[key] = feat_path

            progress.advance(task, len(batch_keys))

    logger.info("Extracted features for %d sites -> %s", len(saved), output_dir)
    return saved


@torch.no_grad()
def extract_plate_features(
    image_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    batch_size: int = 32,
    save_tensors: bool = True,
    tensor_output_dir: Path | None = None,
    tensor_size: int | None = None,
) -> dict[ImageKey, Path]:
    """Extract DINOv3 features for an entire plate.

    Loads a fresh model then delegates to
    :func:`extract_plate_features_with_model`. See that function for
    full parameter documentation.
    """
    model, processor = load_dinov3(model_name, device)
    return extract_plate_features_with_model(
        image_dir,
        output_dir,
        model,
        processor,
        device=device,
        batch_size=batch_size,
        save_tensors=save_tensors,
        tensor_output_dir=tensor_output_dir,
        tensor_size=tensor_size,
    )


def verify_plate_features(
    feature_dir: Path,
    image_dir: Path,
) -> tuple[int, int, list[ImageKey]]:
    """Verify feature extraction completeness for a plate.

    Args:
        feature_dir: Directory containing ``.pt`` feature files.
        image_dir: Original plate ``Images/`` directory.

    Returns:
        ``(extracted_count, expected_count, missing_keys)`` tuple.
    """
    expected_sites = discover_sites(image_dir)
    expected_keys = set(expected_sites.keys())

    existing_files = set(feature_dir.glob("*.pt"))
    existing_names = {f.stem for f in existing_files}

    missing = []
    for key in sorted(expected_keys, key=str):
        if str(key) not in existing_names:
            missing.append(key)

    return len(existing_files), len(expected_keys), missing
