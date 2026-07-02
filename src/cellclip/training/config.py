"""Config loading for local CellCLIP training."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchmark.data import normalize_subset_label


@dataclass(slots=True)
class CellCLIPDatasetConfig:
    """Dataset and split settings for local CellCLIP training."""

    dataset_config_path: str = "configs/dataset.yml"
    feature_root: str = "data/features_cellclip_base"
    split_strategy: str = "cpjump1_official_representation"
    split_manifest_path: str | None = None
    subset: str = "train"
    eval_subset: str = "validate"
    text_level: str = "full"
    exclude_controls: bool = True
    unique_perturbations: bool = False
    max_sites_per_well: int | None = None
    train_max_sites_per_well: int | None = None
    eval_max_sites_per_well: int | None = None
    within_well_interp_sites: int = 0
    same_pert_interp_sites: int = 0
    interp_alpha: float = 0.4
    batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    max_train_wells: int | None = None
    max_eval_wells: int | None = None


@dataclass(slots=True)
class CellCLIPModelConfig:
    """Model settings for CellCLIP training."""

    variant: str = "baseline"
    embed_dim: int = 512
    vision_layers: int = 12
    vision_width: int = 1536
    vision_heads: int = 8
    input_channels: int = 5
    context_length: int = 512
    pooling: str = "attention"
    text_model_name: str = "bert-base-cased"
    tokenizer_name: str = "bert-base-cased"
    chemberta_model_name: str = "seyonec/PubChem10M_SMILES_BPE_450k"
    chemberta_tokenizer_name: str = "seyonec/PubChem10M_SMILES_BPE_450k"
    chemberta_context_length: int = 512
    chemberta_pooling: str = "mean"
    chem_fusion_type: str = "film"
    chem_prompt_policy: str = "remove_smiles"
    freeze_chemberta: bool = True
    chemberta_tune_layers: int = 0
    use_bias: bool = False


@dataclass(slots=True)
class CellCLIPOptimizationConfig:
    """Optimizer and scheduler settings."""

    loss_type: str = "cwcl"
    lr: float = 5.0e-4
    weight_decay: float = 0.2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    epochs: int = 10
    warmup_steps: int = 100
    grad_clip_norm: float = 20.0


@dataclass(slots=True)
class CellCLIPRuntimeConfig:
    """Runtime and output settings."""

    seed: int = 42
    device: str = "auto"
    amp: bool = True
    output_root: str = "output/train_runs"
    run_name: str | None = None
    log_every_steps: int = 10
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    max_train_steps: int | None = None


@dataclass(slots=True)
class CellCLIPTensorBoardConfig:
    """TensorBoard logging settings for CellCLIP."""

    enabled: bool = True
    histogram_every_epochs: int = 5
    flush_every_steps: int = 50


@dataclass(slots=True)
class CellCLIPDistributedConfig:
    """Distributed training settings (multi-GPU via torchrun)."""

    enabled: bool = False
    backend: str = "nccl"
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False


@dataclass(slots=True)
class CellCLIPTrainingConfig:
    """Top-level training config."""

    dataset: CellCLIPDatasetConfig = field(default_factory=CellCLIPDatasetConfig)
    model: CellCLIPModelConfig = field(default_factory=CellCLIPModelConfig)
    optimization: CellCLIPOptimizationConfig = field(default_factory=CellCLIPOptimizationConfig)
    runtime: CellCLIPRuntimeConfig = field(default_factory=CellCLIPRuntimeConfig)
    tensorboard: CellCLIPTensorBoardConfig = field(default_factory=CellCLIPTensorBoardConfig)
    distributed: CellCLIPDistributedConfig = field(default_factory=CellCLIPDistributedConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a YAML-friendly dict."""
        return asdict(self)


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
    for key, value in raw.items():
        if not hasattr(instance, key):
            raise ValueError(f"Unknown config key: {type(instance).__name__}.{key}")
        setattr(instance, key, value)
    return instance


def load_training_config(path: str | Path) -> CellCLIPTrainingConfig:
    """Load a training config YAML file."""
    path = Path(path)
    raw = _load_raw_config(path)

    config = CellCLIPTrainingConfig()
    _merge_dataclass(config.dataset, raw.get("dataset", {}))
    _merge_dataclass(config.model, raw.get("model", {}))
    _merge_dataclass(config.optimization, raw.get("optimization", {}))
    _merge_dataclass(config.runtime, raw.get("runtime", {}))
    _merge_dataclass(config.tensorboard, raw.get("tensorboard", {}))
    _merge_dataclass(config.distributed, raw.get("distributed", {}))

    config.dataset.subset = normalize_subset_label(config.dataset.subset)
    config.dataset.eval_subset = normalize_subset_label(config.dataset.eval_subset)
    config.model.variant = str(config.model.variant).strip().lower()
    if config.model.variant not in {"baseline", "chemberta_film", "chemberta"}:
        raise ValueError(f"Unsupported model.variant: {config.model.variant!r}")
    config.model.chemberta_pooling = str(config.model.chemberta_pooling).strip().lower()
    if config.model.chemberta_pooling not in {"mean", "cls"}:
        raise ValueError("model.chemberta_pooling must be one of {'mean', 'cls'}")
    config.model.chem_fusion_type = str(config.model.chem_fusion_type).strip().lower()
    if config.model.chem_fusion_type not in {"film", "residual_add", "concat_mlp"}:
        raise ValueError(
            "model.chem_fusion_type must be one of {'film', 'residual_add', 'concat_mlp'}"
        )
    config.model.chem_prompt_policy = str(config.model.chem_prompt_policy).strip().lower()
    if config.model.chem_prompt_policy not in {"remove_smiles", "keep_smiles"}:
        raise ValueError("model.chem_prompt_policy must be one of {'remove_smiles', 'keep_smiles'}")
    config.model.chemberta_tune_layers = int(config.model.chemberta_tune_layers)
    if config.model.chemberta_tune_layers < 0:
        raise ValueError("model.chemberta_tune_layers must be >= 0")
    if config.model.freeze_chemberta and config.model.chemberta_tune_layers > 0:
        raise ValueError(
            "model.chemberta_tune_layers must be 0 when model.freeze_chemberta is true"
        )
    if config.model.variant == "chemberta_film":
        config.model.chem_fusion_type = "film"
    if not isinstance(config.optimization.betas, tuple):
        config.optimization.betas = tuple(config.optimization.betas)
    if len(config.optimization.betas) != 2:
        raise ValueError("optimization.betas must contain exactly two floats")
    for field_name in (
        "max_sites_per_well",
        "train_max_sites_per_well",
        "eval_max_sites_per_well",
    ):
        value = getattr(config.dataset, field_name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"dataset.{field_name} must be > 0 when set")
    for field_name in ("within_well_interp_sites", "same_pert_interp_sites"):
        value = int(getattr(config.dataset, field_name))
        setattr(config.dataset, field_name, value)
        if value < 0:
            raise ValueError(f"dataset.{field_name} must be >= 0")
    config.dataset.interp_alpha = float(config.dataset.interp_alpha)
    if config.dataset.interp_alpha <= 0:
        raise ValueError("dataset.interp_alpha must be > 0")
    return config
