"""Configuration resolution for the stable CPJUMP1 benchmark.

Pure config layer: merges CLI overrides with the YAML ``benchmark`` section into
a single :class:`BenchmarkParams`. Depends only on pandas/yaml and
``benchmark.data`` (no copairs/scikit-learn), so it is unit-testable without the
optional ``benchmark`` extra.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from benchmark.data import load_split_manifest, normalize_subset_label

CONFIG_PATH = Path("configs/benchmark.yml")
TIMELINE_CHOICES = ("short", "long")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_benchmark_config(path: Path) -> dict:
    """Load benchmark defaults from YAML if present."""
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark config must be a mapping: {path}")

    config = raw.get("benchmark", raw)
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark config section must be a mapping: {path}")
    return config


def normalize_timelines(value: Any) -> list[str]:
    """Normalize timeline selection to a validated list of labels."""
    if value is None:
        return list(TIMELINE_CHOICES)

    values = [value] if isinstance(value, str) else list(value)

    normalized: list[str] = []
    for item in values:
        label = str(item).strip().lower()
        if label not in TIMELINE_CHOICES:
            choices = ", ".join(TIMELINE_CHOICES)
            raise ValueError(f"Invalid timeline {item!r}; expected one of: {choices}")
        if label not in normalized:
            normalized.append(label)

    return normalized or list(TIMELINE_CHOICES)


def resolve_path(project_root: Path, value: str | None) -> Path | None:
    """Resolve an optional CLI/config path relative to the project root."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


@dataclass(frozen=True)
class BenchmarkParams:
    """Fully-resolved benchmark run parameters (overrides merged with config)."""

    project_root: Path
    profiles_path: Path
    output_path: Path
    figures_dir: Path
    tables_dir: Path
    batch: str
    test_mode: bool
    cell_filter: str | None
    batch_correction: bool
    pca_kernel: str
    pca_n_components: int
    timelines: list[str]
    split_manifest_path: Path | None
    subset: str | None
    manifest: pd.DataFrame | None


def resolve_params(
    *,
    config: Path,
    profiles_dir: str | None,
    output_dir: str | None,
    batch: str | None,
    test_mode: bool | None,
    cell_filter: str | None,
    batch_correction: bool | None,
    pca_kernel: str | None,
    pca_n_components: int | None,
    split_manifest: str | None,
    split_subset: str | None,
    timelines: list[str] | None,
) -> BenchmarkParams:
    """Merge CLI overrides with the YAML config into a :class:`BenchmarkParams`.

    Unset overrides fall back to the ``benchmark`` section of *config*. Loads the
    split manifest when requested.

    Raises:
        ValueError: If ``split_subset``/``split_manifest`` are used inconsistently.
    """
    cfg = load_benchmark_config(config)

    profiles_dir_arg = profiles_dir or cfg.get("profiles_dir", "data/profiles")
    output_dir_arg = output_dir or cfg.get("output_dir", "output/benchmark")
    resolved_batch = batch or cfg.get("batch", "2020_11_04_CPJUMP1")
    resolved_test_mode = test_mode if test_mode is not None else cfg.get("test_mode", False)
    resolved_cell_filter = cell_filter or cfg.get("cell_filter")
    resolved_batch_correction = (
        batch_correction if batch_correction is not None else cfg.get("batch_correction", False)
    )
    resolved_pca_kernel = pca_kernel or cfg.get("pca_kernel", "linear")
    resolved_pca_n_components = int(pca_n_components or cfg.get("pca_n_components", 500))
    split_manifest_arg = split_manifest or cfg.get("split_manifest")
    split_subset_arg = split_subset or cfg.get("split_subset")
    resolved_timelines = normalize_timelines(
        timelines if timelines is not None else cfg.get("timelines")
    )

    project_root = _PROJECT_ROOT
    output_path = project_root / output_dir_arg
    split_manifest_path = resolve_path(project_root, split_manifest_arg)
    subset = None
    manifest = None

    if split_manifest_path is not None:
        if split_subset_arg is None:
            raise ValueError("--split-subset is required when --split-manifest is provided")
        subset = normalize_subset_label(split_subset_arg)
        manifest = load_split_manifest(split_manifest_path, subset)
    elif split_subset_arg is not None:
        raise ValueError("--split-subset requires --split-manifest")

    return BenchmarkParams(
        project_root=project_root,
        profiles_path=project_root / profiles_dir_arg,
        output_path=output_path,
        figures_dir=output_path / "figures",
        tables_dir=output_path / "tables",
        batch=resolved_batch,
        test_mode=resolved_test_mode,
        cell_filter=resolved_cell_filter,
        batch_correction=resolved_batch_correction,
        pca_kernel=resolved_pca_kernel,
        pca_n_components=resolved_pca_n_components,
        timelines=resolved_timelines,
        split_manifest_path=split_manifest_path,
        subset=subset,
        manifest=manifest,
    )
