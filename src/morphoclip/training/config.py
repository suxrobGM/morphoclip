"""Config loading for MorphoCLIP training."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MorphoCLIPDatasetConfig:
    """Dataset, split, and data-loading settings."""

    dataset_config_path: str = "configs/dataset.yml"
    feature_root: str = "data/features"
    text_cache_path: str = "data/text/cached_text_features.pt"
    split_strategy: str = "pert_type"
    text_level: str = "full"
    exclude_controls: bool = True
    max_sites_per_well: int | None = None
    batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    preload: bool = True
    val_fraction: float = 0.1
    max_train_wells: int | None = None
    max_eval_wells: int | None = None


@dataclass(slots=True)
class MorphoCLIPModelConfig:
    """Image encoder and projection head architecture."""

    embed_dim: int = 1024
    output_dim: int = 512
    channel_aggregation: str = "ccf"  # "ccf" (CrossChannelFormer) or "mean_pool"
    ccf_layers: int = 2
    ccf_heads: int = 8
    input_channels: int = 5
    proj_hidden_dim: int = 512
    proj_dropout: float = 0.1
    text_input_dim: int = 768


@dataclass(slots=True)
class MorphoCLIPOptimizationConfig:
    """Optimizer, scheduler, and loss settings."""

    loss_type: str = "infonce"
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    epochs: int = 20
    warmup_steps: int = 200
    grad_clip_norm: float = 1.0
    logit_scale_max: float = 4.6052
    use_cwa: bool = False


@dataclass(slots=True)
class MorphoCLIPRuntimeConfig:
    """Runtime, logging, and checkpointing settings."""

    seed: int = 42
    device: str = "auto"
    amp: bool = True
    output_root: str = "output/morphoclip_runs"
    run_name: str | None = None
    log_every_steps: int = 10
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    max_train_steps: int | None = None


@dataclass(slots=True)
class MorphoCLIPDistributedConfig:
    """Distributed training settings (multi-GPU via torchrun)."""

    enabled: bool = False
    backend: str = "nccl"
    gradient_accumulation_steps: int = 1
    gather_with_grad: bool = True
    find_unused_parameters: bool = False


@dataclass(slots=True)
class TensorBoardConfig:
    """TensorBoard logging settings."""

    histogram_every_epochs: int = 5
    flush_every_steps: int = 50


@dataclass(slots=True)
class MorphoCLIPTrainingConfig:
    """Top-level training config."""

    dataset: MorphoCLIPDatasetConfig = field(
        default_factory=MorphoCLIPDatasetConfig,
    )
    model: MorphoCLIPModelConfig = field(
        default_factory=MorphoCLIPModelConfig,
    )
    optimization: MorphoCLIPOptimizationConfig = field(
        default_factory=MorphoCLIPOptimizationConfig,
    )
    runtime: MorphoCLIPRuntimeConfig = field(
        default_factory=MorphoCLIPRuntimeConfig,
    )
    tensorboard: TensorBoardConfig = field(
        default_factory=TensorBoardConfig,
    )
    distributed: MorphoCLIPDistributedConfig = field(
        default_factory=MorphoCLIPDistributedConfig,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a YAML-friendly dict."""
        return asdict(self)


# --- YAML loading with ``extends`` inheritance ---


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config with optional recursive ``extends`` support."""
    resolved = path.resolve()
    if seen is None:
        seen = set()
    if resolved in seen:
        raise ValueError(f"Config extends cycle detected at {path}")
    seen.add(resolved)

    with open(resolved, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a mapping: {path}")

    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    if isinstance(extends, (str, Path)):
        parent_paths = [Path(extends)]
    elif isinstance(extends, list):
        parent_paths = [Path(item) for item in extends]
    else:
        raise ValueError("Config 'extends' must be a path or list of paths")

    merged: dict[str, Any] = {}
    for parent in parent_paths:
        parent_path = parent if parent.is_absolute() else resolved.parent / parent
        merged = _merge_dicts(merged, _load_raw_config(parent_path, seen.copy()))

    return _merge_dicts(merged, raw)


def _merge_dataclass(instance: Any, raw: dict[str, Any]) -> Any:
    """Merge raw dict values into a dataclass instance."""
    for key, value in raw.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key: {type(instance).__name__}.{key}")
        setattr(instance, key, value)
    return instance


def load_training_config(path: str | Path) -> MorphoCLIPTrainingConfig:
    """Load a MorphoCLIP training config from a YAML file.

    Supports ``extends`` field for config inheritance.

    Args:
        path: Path to the YAML config file.

    Returns:
        Populated training config.
    """
    path = Path(path)
    raw = _load_raw_config(path)

    config = MorphoCLIPTrainingConfig()
    _merge_dataclass(config.dataset, raw.get("dataset", {}))
    _merge_dataclass(config.model, raw.get("model", {}))
    _merge_dataclass(config.optimization, raw.get("optimization", {}))
    _merge_dataclass(config.runtime, raw.get("runtime", {}))
    _merge_dataclass(config.tensorboard, raw.get("tensorboard", {}))
    _merge_dataclass(config.distributed, raw.get("distributed", {}))

    if not isinstance(config.optimization.betas, tuple):
        config.optimization.betas = tuple(config.optimization.betas)
    if len(config.optimization.betas) != 2:
        raise ValueError("optimization.betas must contain exactly two floats")

    return config
