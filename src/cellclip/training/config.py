"""Config schema for local CellCLIP training."""

from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import Field, field_validator, model_validator

from benchmark.data import normalize_subset_label
from morphoclip.config import StrictModel, load_config

ChemFusionType = Literal["film", "residual_add", "concat_mlp"]
ModelVariant = Literal["baseline", "chemberta_film", "chemberta"]

PositiveInt = Annotated[int, Field(gt=0)]

# Fields written as free text in YAML but compared against a fixed set in code.
_CASE_INSENSITIVE = ("variant", "chemberta_pooling", "chem_fusion_type", "chem_prompt_policy")


class CellCLIPDatasetConfig(StrictModel):
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
    max_sites_per_well: PositiveInt | None = None
    train_max_sites_per_well: PositiveInt | None = None
    eval_max_sites_per_well: PositiveInt | None = None
    within_well_interp_sites: int = Field(default=0, ge=0)
    same_pert_interp_sites: int = Field(default=0, ge=0)
    interp_alpha: float = Field(default=0.4, gt=0)
    batch_size: int = 16
    eval_batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    max_train_wells: int | None = None
    max_eval_wells: int | None = None

    @field_validator("subset", "eval_subset")
    @classmethod
    def _normalize_subset(cls, value: str) -> str:
        return normalize_subset_label(value)


class CellCLIPModelConfig(StrictModel):
    """Model settings for CellCLIP training."""

    variant: ModelVariant = "baseline"
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
    chemberta_pooling: Literal["mean", "cls"] = "mean"
    chem_fusion_type: ChemFusionType = "film"
    chem_prompt_policy: Literal["remove_smiles", "keep_smiles"] = "remove_smiles"
    freeze_chemberta: bool = True
    chemberta_tune_layers: int = Field(default=0, ge=0)
    use_bias: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_choices(cls, data: Any) -> Any:
        """Fold case on the choice fields, then pin the fusion the film variant needs.

        Runs before field validation so an out-of-range value still reports its
        own name rather than being silently overwritten.
        """
        if not isinstance(data, dict):
            return data
        normalized = {
            key: value.strip().lower()
            if key in _CASE_INSENSITIVE and isinstance(value, str)
            else value
            for key, value in data.items()
        }
        fusion = normalized.get("chem_fusion_type", "film")
        if normalized.get("variant") == "chemberta_film" and fusion in get_args(ChemFusionType):
            normalized["chem_fusion_type"] = "film"
        return normalized

    @model_validator(mode="after")
    def _tuning_requires_an_unfrozen_encoder(self) -> Self:
        if self.freeze_chemberta and self.chemberta_tune_layers > 0:
            raise ValueError("chemberta_tune_layers must be 0 when freeze_chemberta is true")
        return self


class CellCLIPOptimizationConfig(StrictModel):
    """Optimizer and scheduler settings."""

    loss_type: str = "cwcl"
    lr: float = 5.0e-4
    weight_decay: float = 0.2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    epochs: int = 10
    warmup_steps: int = 100
    grad_clip_norm: float = 20.0


class CellCLIPRuntimeConfig(StrictModel):
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


class CellCLIPTensorBoardConfig(StrictModel):
    """TensorBoard logging settings for CellCLIP."""

    enabled: bool = True
    histogram_every_epochs: int = 5


class CellCLIPDistributedConfig(StrictModel):
    """Distributed training settings (multi-GPU via torchrun)."""

    enabled: bool = False
    backend: str = "nccl"
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False


class CellCLIPTrainingConfig(StrictModel):
    """Top-level training config."""

    dataset: CellCLIPDatasetConfig = Field(default_factory=CellCLIPDatasetConfig)
    model: CellCLIPModelConfig = Field(default_factory=CellCLIPModelConfig)
    optimization: CellCLIPOptimizationConfig = Field(default_factory=CellCLIPOptimizationConfig)
    runtime: CellCLIPRuntimeConfig = Field(default_factory=CellCLIPRuntimeConfig)
    tensorboard: CellCLIPTensorBoardConfig = Field(default_factory=CellCLIPTensorBoardConfig)
    distributed: CellCLIPDistributedConfig = Field(default_factory=CellCLIPDistributedConfig)


def load_training_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> CellCLIPTrainingConfig:
    """Load a CellCLIP training config from YAML, applying ``--set`` overrides."""
    return load_config(CellCLIPTrainingConfig, path, overrides)
