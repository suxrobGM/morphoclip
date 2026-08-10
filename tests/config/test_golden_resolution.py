"""Pin what every committed config resolves to.

`extends` inheritance runs up to six levels deep, so "which value wins" is not
readable from any single file. These goldens record the answer for every config
in the repo, which is what makes it safe to rewrite the loader underneath.

Regenerate deliberately, never as a reflex::

    MORPHOCLIP_REGEN_GOLDENS=1 uv run pytest tests/config/test_golden_resolution.py

A diff here means resolution changed. That is either the point of your change or
a bug, and only you know which.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cellclip.training.config import load_training_config as load_cellclip_config
from morphoclip.training.config import load_training_config as load_morphoclip_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "goldens"
REGEN = os.environ.get("MORPHOCLIP_REGEN_GOLDENS") == "1"

TRAIN_CONFIGS = sorted((PROJECT_ROOT / "configs" / "train").glob("*.yaml"))
CELLCLIP_CONFIGS = sorted((PROJECT_ROOT / "configs" / "cellclip").glob("*.yaml"))

CASES: list[tuple[Path, Callable[[Path], Any]]] = [
    *((path, load_morphoclip_config) for path in TRAIN_CONFIGS),
    *((path, load_cellclip_config) for path in CELLCLIP_CONFIGS),
]


def _golden_name(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}.json"


@pytest.mark.parametrize(
    ("config_path", "loader"),
    CASES,
    ids=[f"{path.parent.name}/{path.stem}" for path, _ in CASES],
)
def test_config_resolves_to_golden(config_path: Path, loader: Callable[[Path], Any]) -> None:
    # Round-trip through JSON so tuples and lists compare equal either way.
    resolved = json.loads(json.dumps(loader(config_path).to_dict(), sort_keys=True))
    golden_path = GOLDEN_DIR / _golden_name(config_path)

    if REGEN:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.skip(f"Recorded {golden_path.name}")

    assert golden_path.exists(), (
        f"No golden for {config_path}. Record it with "
        f"MORPHOCLIP_REGEN_GOLDENS=1 uv run pytest {Path(__file__).name}"
    )
    assert resolved == json.loads(golden_path.read_text(encoding="utf-8"))


def test_no_orphan_goldens() -> None:
    """A deleted config must take its golden with it."""
    expected = {_golden_name(path) for path, _ in CASES}
    assert {path.name for path in GOLDEN_DIR.glob("*.json")} == expected
