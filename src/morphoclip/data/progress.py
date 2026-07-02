"""Pipeline progress tracking: dataclasses, persistence, and status constants.

Tracks per-plate extraction state in a JSON file for crash-safe resume.
Used by :class:`~morphoclip.data.pipeline.PlateExtractionPipeline`.
"""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_MAX_SITES_PER_PLATE = 3456
_COMPLETENESS_THRESHOLD = 0.80


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class PlateStatus:
    """Valid status values for a plate in the progress file."""

    PENDING = "pending"
    SKIPPED = "skipped"
    FETCHING = "fetching"
    EXTRACTING = "extracting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlateRecord:
    """Progress record for a single plate."""

    status: str = PlateStatus.PENDING
    barcode: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    sites_extracted: int = 0
    error: str | None = None


@dataclass
class PipelineProgress:
    """Full pipeline progress state, serialized to JSON."""

    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    config_hash: str = ""
    metadata_downloaded: bool = False
    plates: dict[str, dict[str, Any]] = field(default_factory=dict)


def compute_config_hash(plates: list[str]) -> str:
    """SHA-256 of the sorted plate list."""
    plates_str = json.dumps(sorted(plates))
    return f"sha256:{hashlib.sha256(plates_str.encode()).hexdigest()[:16]}"


def load_progress(path: Path, config_hash: str) -> PipelineProgress:
    """Load existing progress JSON or initialize fresh.

    In-progress plates from a previous crash (FETCHING/EXTRACTING) are
    automatically reset to PENDING.

    Args:
        path: Path to the progress JSON file.
        config_hash: Current config hash for change detection.

    Returns:
        Loaded or fresh ``PipelineProgress`` instance.
    """
    if not path.exists():
        return PipelineProgress(config_hash=config_hash)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        progress = PipelineProgress(
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            created_at=raw.get("created_at", _utcnow()),
            updated_at=raw.get("updated_at", _utcnow()),
            config_hash=raw.get("config_hash", ""),
            metadata_downloaded=raw.get("metadata_downloaded", False),
            plates=raw.get("plates", {}),
        )

        if progress.config_hash and progress.config_hash != config_hash:
            logger.warning(
                "Config hash changed (plates list may have been edited). "
                "Completed plates will not be reprocessed."
            )
        progress.config_hash = config_hash

        # Reset in-progress plates from a previous crash
        for plate_name, record in progress.plates.items():
            if record["status"] in {PlateStatus.FETCHING, PlateStatus.EXTRACTING}:
                logger.info("Resetting crashed plate %s to pending", plate_name)
                record["status"] = PlateStatus.PENDING
                record["error"] = None

        return progress

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Corrupt progress file, starting fresh: %s", exc)
        backup = path.with_suffix(f".json.corrupt.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(path, backup)
        return PipelineProgress(config_hash=config_hash)


def save_progress(progress: PipelineProgress, path: Path) -> None:
    """Atomically write progress to disk via temp file + os.replace."""
    progress.updated_at = _utcnow()
    data = asdict(progress)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
