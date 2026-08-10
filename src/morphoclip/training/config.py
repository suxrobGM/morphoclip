"""Config schema for MorphoCLIP training."""

from pathlib import Path
from typing import Any

from pydantic import Field

from morphoclip.config import StrictModel, load_config

# Properties of the pre-extracted feature cache, not tunable choices.
EMBED_DIM = 1024  # DINOv3 ViT-L/16 CLS token dimension
TEXT_INPUT_DIM = 768  # BioClinical ModernBERT hidden dimension
INPUT_CHANNELS = 5  # Fluorescence channels (Mito, Actin, Golgi, ER, DNA)
PROJ_HIDDEN_DIM = 512  # ProjectionHead hidden dimension

# Never tuned in practice; fixed to sane defaults.
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1.0e-8
GRAD_CLIP_NORM = 1.0
LOGIT_SCALE_MAX = 4.6052  # ln(100), CLIP default ceiling for learnable temperature


class MorphoCLIPDatasetConfig(StrictModel):
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
    preload: bool = True
    val_fraction: float = 0.1
    # "random" (plain shuffled DataLoader) or "perturbation"
    # (PerturbationBatchSampler: replicate-aware, plate-mixed batches).
    batch_sampler: str = "random"
    replicates_per_group: int = 2  # Target replicates per perturbation chunk


class MorphoCLIPModelConfig(StrictModel):
    """Image encoder and projection head architecture."""

    output_dim: int = 512
    # Well aggregation: "ccf-mean", "meanpool-mean", "ccf-attn", "wellformer".
    aggregator: str = "ccf-mean"
    # Transformer depth/width, shared by CrossChannelFormer and WellFormer.
    ccf_layers: int = 2
    ccf_heads: int = 8
    proj_dropout: float = 0.1


class MorphoCLIPOptimizationConfig(StrictModel):
    """Optimizer, scheduler, and loss settings."""

    loss_type: str = "infonce"
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    epochs: int = 20
    warmup_steps: int = 200
    use_cwa: bool = False
    # Gene-aware CWCL: affinity for pairs whose target genes intersect but
    # whose broad_sample differs. 0.0 gives binary labels.
    target_weight: float = 0.0
    # Weight of the replicate image-image alignment term (0.0 disables).
    replicate_weight: float = 0.0
    # Fixed temperature for the replicate term; None reuses the shared
    # learnable logit scale.
    replicate_temperature: float | None = None


class MorphoCLIPRuntimeConfig(StrictModel):
    """Runtime, logging, and checkpointing settings."""

    seed: int = 42
    device: str = "auto"
    amp: bool = True
    output_root: str = "output/morphoclip_runs"
    run_name: str | None = None
    log_every_steps: int = 10
    max_train_steps: int | None = None
    # Epochs without a new best eval loss before stopping (None disables).
    early_stop_patience: int | None = None


class MorphoCLIPDistributedConfig(StrictModel):
    """Distributed training settings (multi-GPU via torchrun)."""

    enabled: bool = False
    gradient_accumulation_steps: int = 1


class MorphoCLIPTrainingConfig(StrictModel):
    """Top-level training config."""

    dataset: MorphoCLIPDatasetConfig = Field(default_factory=MorphoCLIPDatasetConfig)
    model: MorphoCLIPModelConfig = Field(default_factory=MorphoCLIPModelConfig)
    optimization: MorphoCLIPOptimizationConfig = Field(default_factory=MorphoCLIPOptimizationConfig)
    runtime: MorphoCLIPRuntimeConfig = Field(default_factory=MorphoCLIPRuntimeConfig)
    distributed: MorphoCLIPDistributedConfig = Field(default_factory=MorphoCLIPDistributedConfig)


def training_config_from_dict(config_dict: dict[str, Any]) -> MorphoCLIPTrainingConfig:
    """Rebuild a training config from a plain nested dict, e.g. from a checkpoint."""
    return MorphoCLIPTrainingConfig.model_validate(config_dict)


def load_training_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> MorphoCLIPTrainingConfig:
    """Load a MorphoCLIP training config from YAML, applying ``--set`` overrides."""
    return load_config(MorphoCLIPTrainingConfig, path, overrides)
