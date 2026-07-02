"""Checkpoint loading utilities for the local CellCLIP runtime."""

from pathlib import Path

import torch

from cellclip.benchmark.model import CellCLIPVisualConfig, CellCLIPVisualEncoder

DEFAULT_CHECKPOINT_REPO = "suinleelab/CellCLIP"
DEFAULT_CHECKPOINT_FILENAME = "model.safetensors"


def resolve_checkpoint(
    ckpt_path: str | None,
    checkpoint_repo_id: str,
    checkpoint_filename: str,
    download_dir: Path | None,
) -> str:
    """Resolve checkpoint path from local input or Hugging Face."""
    if ckpt_path:
        return str(Path(ckpt_path).expanduser().resolve())

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise RuntimeError(
            "huggingface_hub is required to download the default CellCLIP checkpoint. "
            "Install it or pass --ckpt-path."
        ) from exc

    kwargs = {
        "repo_id": checkpoint_repo_id,
        "filename": checkpoint_filename,
    }
    if download_dir is not None:
        kwargs["local_dir"] = str(download_dir)

    return hf_hub_download(**kwargs)


def _load_raw_checkpoint(checkpoint_path: str, device: str) -> dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError as exc:  # pragma: no cover - depends on runtime environment
            raise RuntimeError(
                "safetensors is required to load the default CellCLIP checkpoint format."
            ) from exc
        raw = safe_load_file(checkpoint_path, device=device)
    else:
        raw = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return raw["model"]
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {checkpoint_path}")
    return raw


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    prefix_with_dot = f"{prefix}."
    if not any(key.startswith(prefix_with_dot) for key in state_dict):
        return state_dict
    return {
        key[len(prefix_with_dot) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix_with_dot)
    }


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    if any(key.startswith("visual.") for key in state_dict):
        return state_dict

    stripped = _strip_prefix(state_dict, "visual")
    if stripped is not state_dict:
        return {f"visual.{key}": value for key, value in stripped.items()}

    return state_dict


def _validate_visual_width(state_dict: dict[str, torch.Tensor], input_dim: int) -> None:
    channel_embed = state_dict.get("visual.channel_embed")
    if channel_embed is None:
        raise RuntimeError("Checkpoint is missing required key: visual.channel_embed")

    checkpoint_dim = int(channel_embed.shape[-1])
    if checkpoint_dim != input_dim:
        raise RuntimeError(
            f"Checkpoint visual width is {checkpoint_dim}, but --input-dim was {input_dim}"
        )


def load_cellclip_visual_encoder(
    model_path: str,
    device: str,
    input_dim: int,
    embed_dim: int = 512,
    vision_layers: int = 12,
    vision_heads: int = 8,
    input_channels: int = 5,
    pooling: str = "attention",
) -> CellCLIPVisualEncoder:
    """Load the image-encoding path of a CellCLIP checkpoint."""
    state_dict = _normalize_state_dict(_load_raw_checkpoint(model_path, device))
    _validate_visual_width(state_dict, input_dim)

    model = CellCLIPVisualEncoder(
        CellCLIPVisualConfig(
            embed_dim=embed_dim,
            vision_layers=vision_layers,
            vision_width=input_dim,
            vision_heads=vision_heads,
            input_channels=input_channels,
            pooling=pooling,
        )
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    required_missing = [
        key for key in missing if key.startswith("visual.") or key.startswith("image_pool.")
    ]
    if required_missing:
        preview = ", ".join(required_missing[:5])
        raise RuntimeError(f"Checkpoint is missing visual pooling weights: {preview}")

    unexpected_visual = [
        key for key in unexpected if key.startswith("visual.") or key.startswith("image_pool.")
    ]
    if unexpected_visual:
        preview = ", ".join(unexpected_visual[:5])
        raise RuntimeError(f"Checkpoint has unexpected visual pooling weights: {preview}")

    if device == "cpu":
        model.float()

    model.to(device)
    model.eval()
    return model
