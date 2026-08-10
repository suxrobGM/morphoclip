"""Shared configuration primitives: strict models and YAML ``extends`` loading.

Every config in this repo is a nested mapping of typed sections, read from a YAML
file that may ``extends`` one or more parents and then be overridden by
repeatable ``section.key=value`` CLI flags. This module is the one implementation
of that.

``extends`` is resolved here rather than in a model validator, for two reasons:
a validator has no access to the file it came from, and a config rebuilt from a
checkpoint must never re-read a parent file that may have changed since.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Base for config sections. Unknown keys fail loudly; assignment is checked."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain nested dict for YAML dumps and checkpoints."""
        return self.model_dump()


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* onto *base*, returning a new dict."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file that must hold a mapping. An empty file gives ``{}``.

    Raises:
        ValueError: If the file parses to something other than a mapping.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return raw


def resolve_extends(path: str | Path, _ancestors: frozenset[Path] = frozenset()) -> dict[str, Any]:
    """Load a YAML config with its ``extends`` parents merged in.

    Parents are merged left to right, then the child wins over all of them.

    Args:
        path: Path to the YAML file.
        _ancestors: Files already open further up the chain, used for cycle detection.

    Returns:
        The fully merged mapping, with ``extends`` itself removed.

    Raises:
        ValueError: On an ``extends`` cycle, or an ``extends`` value that is not
            a path or list of paths.
    """
    resolved = Path(path).resolve()
    if resolved in _ancestors:
        raise ValueError(f"Config extends cycle detected at {path}")

    raw = load_yaml_mapping(resolved)
    extends = raw.pop("extends", None)
    if extends is None:
        return raw

    if isinstance(extends, (str, Path)):
        parents = [Path(extends)]
    elif isinstance(extends, list):
        parents = [Path(item) for item in extends]
    else:
        raise ValueError("Config 'extends' must be a path or list of paths")

    ancestors = _ancestors | {resolved}
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = parent if parent.is_absolute() else resolved.parent / parent
        merged = merge_dicts(merged, resolve_extends(parent_path, ancestors))
    return merge_dicts(merged, raw)


def parse_dotted_override(item: str) -> dict[str, Any]:
    """Parse one ``section.key=value`` CLI override into a nested dict.

    The value goes through :func:`yaml.safe_load`, so ints, floats, bools and
    null round-trip without extra quoting.

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

    nested: Any = yaml.safe_load(raw_value)
    for part in reversed(parts[1:]):
        nested = {part: nested}
    return {parts[0]: nested}


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides, in order, onto a resolved mapping."""
    for item in overrides or []:
        raw = merge_dicts(raw, parse_dotted_override(item))
    return raw


def load_config[ModelT: BaseModel](
    model: type[ModelT],
    path: str | Path,
    overrides: Iterable[str] | None = None,
) -> ModelT:
    """Resolve ``extends``, apply ``--set`` overrides, and validate into *model*.

    Args:
        model: The config model to validate into.
        path: Path to the YAML config file.
        overrides: ``section.key=value`` strings from repeatable ``--set`` options.

    Returns:
        A validated instance of *model*.

    Raises:
        pydantic.ValidationError: If the resolved mapping does not match the schema.
    """
    return model.model_validate(apply_overrides(resolve_extends(path), overrides))
