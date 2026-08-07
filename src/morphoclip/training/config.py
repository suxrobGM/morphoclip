"""Config loading for MorphoCLIP training."""

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# --- Model constants ---
# Properties of the pre-extracted feature cache, not tunable choices.
EMBED_DIM = 1024  # DINOv3 ViT-L/16 CLS token dimension
TEXT_INPUT_DIM = 768  # BioClinical ModernBERT hidden dimension
INPUT_CHANNELS = 5  # Fluorescence channels (Mito, Actin, Golgi, ER, DNA)
PROJ_HIDDEN_DIM = 512  # ProjectionHead hidden dimension

# --- Optimizer constants ---
# Never tuned in practice; fixed to sane defaults.
ADAMW_BETAS = (0.9, 0.999)
ADAMW_EPS = 1.0e-8
GRAD_CLIP_NORM = 1.0
LOGIT_SCALE_MAX = 4.6052  # ln(100), CLIP default ceiling for learnable temperature


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
    # NOTE: kept as an explicit field (not derived from batch_size) because
    # morphoclip.benchmark.export._encode_plate_wells assigns
    # ``config.dataset.eval_batch_size`` directly to override the per-plate
    # export batch size; that module is out of scope for this refactor.
    eval_batch_size: int = 32
    preload: bool = True
    val_fraction: float = 0.1
    # "random" (plain shuffled DataLoader) or "perturbation"
    # (PerturbationBatchSampler: replicate-aware, plate-mixed batches).
    batch_sampler: str = "random"
    replicates_per_group: int = 2  # Target replicates per perturbation chunk


@dataclass(slots=True)
class MorphoCLIPModelConfig:
    """Image encoder and projection head architecture."""

    output_dim: int = 512
    # Well aggregation: "ccf-mean", "meanpool-mean", "ccf-attn", "wellformer".
    aggregator: str = "ccf-mean"
    # Transformer depth/width, shared by CrossChannelFormer and WellFormer.
    ccf_layers: int = 2
    ccf_heads: int = 8
    proj_dropout: float = 0.1


@dataclass(slots=True)
class MorphoCLIPOptimizationConfig:
    """Optimizer, scheduler, and loss settings."""

    loss_type: str = "infonce"
    lr: float = 3.0e-4
    weight_decay: float = 0.1
    epochs: int = 20
    warmup_steps: int = 200
    use_cwa: bool = False
    # Gene-aware CWCL: affinity given to pairs whose target gene sets
    # intersect but whose broad_sample differs.  0.0 disables (binary labels).
    target_weight: float = 0.0
    # Weight of the replicate image-image alignment term (0.0 disables).
    lambda_img: float = 0.0
    # Fixed temperature for the image-image term; None reuses the shared
    # learnable logit scale.
    img_img_temperature: float | None = None


@dataclass(slots=True)
class MorphoCLIPRuntimeConfig:
    """Runtime, logging, and checkpointing settings."""

    seed: int = 42
    device: str = "auto"
    amp: bool = True
    output_root: str = "output/morphoclip_runs"
    run_name: str | None = None
    log_every_steps: int = 10
    max_train_steps: int | None = None
    # Stop after this many epochs without a new best eval loss.  None
    # disables.  Only applied when not running distributed — eval runs on
    # rank 0 only, so the stop decision is not available on other ranks.
    early_stop_patience: int | None = None


@dataclass(slots=True)
class MorphoCLIPDistributedConfig:
    """Distributed training settings (multi-GPU via torchrun).

    Kept as its own dataclass rather than folded into runtime: ``trainer.py``,
    ``loop.py``, and ``setup.py`` all thread ``config.distributed`` through as
    a distinct namespace (mirroring the ``--distributed`` CLI flag), and
    folding it into runtime would rename call sites across those modules for
    no schema benefit.
    """

    enabled: bool = False
    gradient_accumulation_steps: int = 1


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


def _merge_dataclass(instance: Any, raw: dict[str, Any], *, strict: bool = True) -> Any:
    """Merge raw dict values into a dataclass instance.

    Args:
        instance: Dataclass instance to update in place.
        raw: Mapping of field names to override values.
        strict: If ``True``, raise on keys the dataclass no longer has. If
            ``False``, log a warning and skip them instead — used when
            reconstructing a config from an older checkpoint's embedded dict,
            which may reference fields dropped from the current schema.

    Returns:
        The mutated *instance*.

    Raises:
        ValueError: If *strict* is ``True`` and *raw* contains an unknown key.
    """
    for key, value in raw.items():
        if not hasattr(instance, key):
            if strict:
                raise ValueError(f"Unknown config key: {type(instance).__name__}.{key}")
            logger.warning(
                "Skipping unknown config key %s.%s (likely from an older checkpoint)",
                type(instance).__name__,
                key,
            )
            continue
        setattr(instance, key, value)
    return instance


# Checkpoints saved before the aggregator refactor embed the old
# ``model.channel_aggregation`` field.  Mapping it explicitly keeps those
# checkpoints correct rather than accidentally correct via the default.
LEGACY_CHANNEL_AGGREGATION = {"ccf": "ccf-mean", "mean_pool": "meanpool-mean"}


def _translate_legacy_model_section(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a legacy ``model`` section onto the current schema.

    Translates ``channel_aggregation`` into ``aggregator``. An explicit
    ``aggregator`` already present always wins.

    Args:
        raw: The checkpoint's ``model`` section.

    Returns:
        A copy of *raw* with legacy keys translated and removed.
    """
    if "channel_aggregation" not in raw:
        return raw

    translated = dict(raw)
    legacy = translated.pop("channel_aggregation")
    if "aggregator" in translated:
        return translated

    mapped = LEGACY_CHANNEL_AGGREGATION.get(legacy)
    if mapped is None:
        logger.warning(
            "Unrecognized legacy model.channel_aggregation=%r; using default aggregator",
            legacy,
        )
        return translated

    logger.info("Translating legacy model.channel_aggregation=%r to aggregator=%r", legacy, mapped)
    translated["aggregator"] = mapped
    return translated


def training_config_from_dict(
    config_dict: dict[str, Any],
    *,
    strict: bool = True,
) -> MorphoCLIPTrainingConfig:
    """Reconstruct a training config from a plain nested dict.

    Used both for YAML-loaded configs and for configs embedded in a saved
    checkpoint's ``config`` key.

    Args:
        config_dict: Nested dict with ``dataset``/``model``/``optimization``/
            ``runtime``/``distributed`` sections.
        strict: If ``True``, raise on keys not present in the current schema.
            Pass ``False`` for configs loaded from older checkpoints, whose
            embedded dict may still carry fields the schema has since dropped.
            Non-strict mode also applies legacy key translation (e.g.
            ``model.channel_aggregation`` -> ``model.aggregator``); strict mode
            does not, so stale YAML configs still fail loudly.

    Returns:
        Populated training config.
    """
    config = MorphoCLIPTrainingConfig()
    model_section = config_dict.get("model", {})
    if not strict:
        model_section = _translate_legacy_model_section(model_section)
    _merge_dataclass(config.dataset, config_dict.get("dataset", {}), strict=strict)
    _merge_dataclass(config.model, model_section, strict=strict)
    _merge_dataclass(config.optimization, config_dict.get("optimization", {}), strict=strict)
    _merge_dataclass(config.runtime, config_dict.get("runtime", {}), strict=strict)
    _merge_dataclass(config.distributed, config_dict.get("distributed", {}), strict=strict)
    return config


def _parse_dotted_override(item: str) -> dict[str, Any]:
    """Parse a single ``section.key=value`` CLI override into a nested dict.

    Values are parsed with :func:`yaml.safe_load` so ints/floats/bools/null
    round-trip without extra quoting (e.g. ``runtime.max_train_steps=3``).

    Args:
        item: A ``section.key=value`` string.

    Returns:
        A single-section nested dict, e.g. ``{"runtime": {"max_train_steps": 3}}``.

    Raises:
        ValueError: If *item* has no ``=``, or the key has no ``section.field`` dot.
    """
    if "=" not in item:
        raise ValueError(f"Malformed --set override {item!r}: expected 'section.key=value'")
    dotted_key, raw_value = item.split("=", 1)
    parts = dotted_key.split(".")
    if len(parts) < 2 or not all(parts):
        raise ValueError(
            f"Malformed --set override {item!r}: key must be 'section.key' "
            "(e.g. 'runtime.max_train_steps=3')"
        )

    value = yaml.safe_load(raw_value)
    nested: dict[str, Any] = value
    for part in reversed(parts[1:]):
        nested = {part: nested}
    return {parts[0]: nested}


def load_training_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> MorphoCLIPTrainingConfig:
    """Load a MorphoCLIP training config from a YAML file.

    Supports ``extends`` field for config inheritance, followed by optional
    dotted CLI overrides (``section.key=value``) applied on top.

    Args:
        path: Path to the YAML config file.
        overrides: Optional ``section.key=value`` strings (e.g. from
            repeatable ``--set`` CLI options), applied in order after the
            YAML's own ``extends`` inheritance is resolved.

    Returns:
        Populated training config.
    """
    path = Path(path)
    raw = _load_raw_config(path)
    for item in overrides or []:
        raw = _merge_dicts(raw, _parse_dotted_override(item))

    return training_config_from_dict(raw, strict=True)
