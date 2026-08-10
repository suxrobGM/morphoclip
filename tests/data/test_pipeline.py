"""Tests for morphoclip.data.pipeline and morphoclip.data.progress."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from morphoclip.data.config import CPJumpConfig
from morphoclip.data.pipeline import PlateExtractionPipeline
from morphoclip.data.progress import PlateStatus, _utcnow, compute_config_hash

PLATE = "BR00116991__2020-11-05T19_51_35-Measurement1"
OTHER_PLATE = "BR00116992__2020-11-05T21_31_31-Measurement1"

# int(_MAX_SITES_PER_PLATE * _COMPLETENESS_THRESHOLD) in morphoclip.data.progress:
# the file count at which a plate directory is taken for finished.
COMPLETE_SITES = 2764


def _record(status: str, *, sites: int = 0, error: str | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "barcode": "BR00116991",
        "started_at": _utcnow(),
        "completed_at": None,
        "sites_extracted": sites,
        "error": error,
    }


@pytest.fixture
def progress_path(tmp_path: Path) -> Path:
    return tmp_path / "progress.json"


@pytest.fixture
def make_pipeline(tmp_path: Path, progress_path: Path) -> Callable[..., PlateExtractionPipeline]:
    """Build a dry-run pipeline over two plates, rooted in a temp directory.

    Pass ``progress={plate: record}`` to lay down a progress file first, since
    the pipeline reads it on construction.
    """

    def build(*, progress: dict[str, Any] | None = None, **kwargs: Any) -> PlateExtractionPipeline:
        if progress is not None:
            progress_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_at": _utcnow(),
                        "updated_at": _utcnow(),
                        "config_hash": "",
                        "metadata_downloaded": False,
                        "plates": progress,
                    }
                ),
                encoding="utf-8",
            )
        config = CPJumpConfig(
            batch="2020_11_04_CPJUMP1",
            plates=[PLATE, OTHER_PLATE],
            local={
                "raw_images": f"{tmp_path}/raw",
                "features": f"{tmp_path}/features",
                "tensors": f"{tmp_path}/tensors",
                "metadata": f"{tmp_path}/metadata",
            },
            extraction={"device": "cpu", "batch_size": 4},
            fetch={"rclone_remote": ":s3:"},
        )
        return PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
            **kwargs,
        )

    return build


@pytest.fixture
def pipeline(make_pipeline: Callable[..., PlateExtractionPipeline]) -> PlateExtractionPipeline:
    return make_pipeline()


def _feature_dir(tmp_path: Path, sites: int | None) -> Path:
    """A feature directory holding *sites* empty ``.pt`` files, or missing entirely."""
    feature_dir = tmp_path / "features" / "BR00116991"
    if sites is None:
        return feature_dir
    feature_dir.mkdir(parents=True)
    for i in range(sites):
        (feature_dir / f"site_{i:05d}.pt").touch()
    return feature_dir


def test_a_fresh_save_writes_the_schema_and_leaves_no_temp_file(
    pipeline: PlateExtractionPipeline, progress_path: Path
) -> None:
    assert not progress_path.exists()

    pipeline._save_progress()

    assert list(progress_path.parent.glob("*.tmp")) == []
    saved = json.loads(progress_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert saved["metadata_downloaded"] is False
    assert saved["config_hash"] == compute_config_hash([PLATE, OTHER_PLATE])


def test_a_corrupt_progress_file_is_backed_up_and_the_run_starts_fresh(
    make_pipeline: Callable[..., PlateExtractionPipeline], progress_path: Path
) -> None:
    progress_path.write_text("not json{{{", encoding="utf-8")

    pipeline = make_pipeline()

    assert pipeline._progress.metadata_downloaded is False
    assert pipeline._progress.plates == {}
    assert len(list(progress_path.parent.glob("*.corrupt.*"))) == 1


def test_plates_left_mid_run_are_reset_to_pending_on_load(
    make_pipeline: Callable[..., PlateExtractionPipeline],
) -> None:
    plates = make_pipeline(
        progress={
            PLATE: _record(PlateStatus.FETCHING, error="boom"),
            OTHER_PLATE: _record(PlateStatus.EXTRACTING),
        }
    )._progress.plates

    assert [plates[name]["status"] for name in (PLATE, OTHER_PLATE)] == [PlateStatus.PENDING] * 2
    assert plates[PLATE]["error"] is None


@pytest.mark.parametrize(
    ("sites", "complete"),
    [(None, False), (0, False), (COMPLETE_SITES - 1, False), (COMPLETE_SITES, True)],
)
def test_without_a_record_completeness_is_a_file_count_heuristic(
    pipeline: PlateExtractionPipeline, tmp_path: Path, sites: int | None, complete: bool
) -> None:
    feature_dir = _feature_dir(tmp_path, sites)
    assert pipeline._check_output_complete(PLATE, feature_dir, tmp_path / "tensors") is complete


@pytest.mark.parametrize(
    ("recorded", "on_disk", "complete"),
    [
        (COMPLETE_SITES, COMPLETE_SITES, True),
        (COMPLETE_SITES, 100, False),
        (100, 100, False),
    ],
)
def test_a_completed_record_is_believed_only_above_the_threshold(
    make_pipeline: Callable[..., PlateExtractionPipeline],
    tmp_path: Path,
    recorded: int,
    on_disk: int,
    complete: bool,
) -> None:
    """A record claiming a short plate is stale, not a smaller completeness bar."""
    feature_dir = _feature_dir(tmp_path, on_disk)
    pipeline = make_pipeline(progress={PLATE: _record(PlateStatus.COMPLETED, sites=recorded)})

    assert pipeline._check_output_complete(PLATE, feature_dir, tmp_path / "tensors") is complete


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, [PLATE, OTHER_PLATE]),
        ([], [PLATE, OTHER_PLATE]),
        (["BR00116991"], [PLATE]),
        ([PLATE], [PLATE]),
        (["NONEXISTENT"], []),
    ],
)
def test_plates_resolve_from_a_barcode_or_a_full_name_and_unknowns_are_dropped(
    pipeline: PlateExtractionPipeline, requested: list[str] | None, expected: list[str]
) -> None:
    assert pipeline._resolve_plates(requested) == expected


def test_retrying_resets_failed_plates_and_clears_their_error(
    make_pipeline: Callable[..., PlateExtractionPipeline],
) -> None:
    failed = _record(PlateStatus.FAILED, error="CalledProcessError: exit code 1")
    pipeline = make_pipeline(progress={PLATE: failed}, retry_failed=True)
    assert pipeline._progress.plates[PLATE]["status"] == PlateStatus.FAILED

    assert pipeline._reset_failed_plates([PLATE, OTHER_PLATE]) == 1

    assert pipeline._progress.plates[PLATE]["status"] == PlateStatus.PENDING
    assert pipeline._progress.plates[PLATE]["error"] is None
    assert pipeline._progress.plates[OTHER_PLATE]["status"] == PlateStatus.PENDING


def test_the_config_hash_ignores_plate_order_but_not_plate_identity() -> None:
    assert compute_config_hash([PLATE, OTHER_PLATE]) == compute_config_hash([OTHER_PLATE, PLATE])
    assert compute_config_hash([PLATE]) != compute_config_hash([OTHER_PLATE])
