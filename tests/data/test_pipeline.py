"""Tests for morphoclip.data.pipeline and morphoclip.data.progress modules."""

import json
from pathlib import Path
from typing import Any

import pytest

from morphoclip.data.pipeline import PlateExtractionPipeline
from morphoclip.data.progress import (
    PipelineProgress,
    PlateRecord,
    PlateStatus,
    _utcnow,
)


def _minimal_config(
    tmp_path: Path | None = None,
    plates: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal cpjump config dict for testing.

    When *tmp_path* is provided, local paths point into the temp directory
    to avoid interference from real project data.
    """
    base = str(tmp_path) if tmp_path else "data"
    return {
        "endpoint": "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4",
        "batch": "2020_11_04_CPJUMP1",
        "images": "images/{batch}/images",
        "metadata": "workspace/metadata/platemaps/{batch}",
        "external_metadata": "workspace/metadata/external_metadata",
        "plates": plates
        or [
            "BR00116991__2020-11-05T19_51_35-Measurement1",
            "BR00116992__2020-11-05T21_31_31-Measurement1",
        ],
        "local": {
            "raw_images": f"{base}/raw",
            "features": f"{base}/features",
            "tensors": f"{base}/tensors",
            "metadata": f"{base}/metadata",
        },
        "extraction": {
            "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "device": "cpu",
            "batch_size": 4,
        },
        "fetch": {
            "backend": "auto",
            "aws_no_sign_request": True,
            "rclone_remote": ":s3:",
        },
    }


@pytest.fixture()
def config(tmp_path: Path) -> dict[str, Any]:
    return _minimal_config(tmp_path)


@pytest.fixture()
def progress_path(tmp_path: Path) -> Path:
    return tmp_path / "progress.json"


class TestPlateRecord:
    def test_defaults(self) -> None:
        record = PlateRecord()
        assert record.status == PlateStatus.PENDING
        assert record.barcode == ""
        assert record.sites_extracted == 0
        assert record.error is None


class TestPipelineProgress:
    def test_defaults(self) -> None:
        progress = PipelineProgress()
        assert progress.schema_version == 1
        assert progress.metadata_downloaded is False
        assert progress.plates == {}


class TestProgressIO:
    def test_fresh_start(self, config: dict, progress_path: Path) -> None:
        """Pipeline creates fresh progress when no file exists."""
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        assert not progress_path.exists()
        pipeline._save_progress()
        assert progress_path.exists()

        data = json.loads(progress_path.read_text())
        assert data["schema_version"] == 1
        assert data["metadata_downloaded"] is False

    def test_load_existing_progress(self, config: dict, progress_path: Path) -> None:
        """Pipeline loads and preserves existing progress."""
        progress_data = {
            "schema_version": 1,
            "created_at": "2026-03-08T00:00:00+00:00",
            "updated_at": "2026-03-08T00:00:00+00:00",
            "config_hash": "",
            "metadata_downloaded": True,
            "plates": {
                "BR00116991__2020-11-05T19_51_35-Measurement1": {
                    "status": "completed",
                    "barcode": "BR00116991",
                    "started_at": "2026-03-08T00:00:00+00:00",
                    "completed_at": "2026-03-08T00:05:00+00:00",
                    "sites_extracted": 3456,
                    "error": None,
                },
            },
        }
        progress_path.write_text(json.dumps(progress_data))

        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        assert pipeline._progress.metadata_downloaded is True
        plate_rec = pipeline._progress.plates["BR00116991__2020-11-05T19_51_35-Measurement1"]
        assert plate_rec["status"] == "completed"
        assert plate_rec["sites_extracted"] == 3456

    def test_corrupt_progress_recovery(self, config: dict, progress_path: Path) -> None:
        """Pipeline recovers from corrupt progress file."""
        progress_path.write_text("not json{{{")

        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        # Should start fresh
        assert pipeline._progress.metadata_downloaded is False
        assert pipeline._progress.plates == {}

        # Corrupt file should be backed up
        backups = list(progress_path.parent.glob("*.corrupt.*"))
        assert len(backups) == 1

    def test_crashed_plates_reset_on_load(self, config: dict, progress_path: Path) -> None:
        """Plates in FETCHING/EXTRACTING status get reset to PENDING on load."""
        progress_data = {
            "schema_version": 1,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "config_hash": "",
            "metadata_downloaded": False,
            "plates": {
                "BR00116991__2020-11-05T19_51_35-Measurement1": {
                    "status": "fetching",
                    "barcode": "BR00116991",
                    "started_at": _utcnow(),
                    "completed_at": None,
                    "sites_extracted": 0,
                    "error": None,
                },
                "BR00116992__2020-11-05T21_31_31-Measurement1": {
                    "status": "extracting",
                    "barcode": "BR00116992",
                    "started_at": _utcnow(),
                    "completed_at": None,
                    "sites_extracted": 0,
                    "error": None,
                },
            },
        }
        progress_path.write_text(json.dumps(progress_data))

        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        for plate_name in progress_data["plates"]:
            assert pipeline._progress.plates[plate_name]["status"] == PlateStatus.PENDING

    def test_atomic_save(self, config: dict, progress_path: Path) -> None:
        """Progress save should not leave .tmp files around."""
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        pipeline._save_progress()

        tmp_files = list(progress_path.parent.glob("*.tmp"))
        assert len(tmp_files) == 0
        assert progress_path.exists()


class TestCheckOutputComplete:
    def test_no_feature_dir(self, config: dict, progress_path: Path, tmp_path: Path) -> None:
        """Returns False when feature dir doesn't exist."""
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        feature_dir = tmp_path / "features" / "BR00116991"
        tensor_dir = tmp_path / "tensors" / "BR00116991"
        assert not pipeline._check_output_complete(
            "BR00116991__2020-11-05T19_51_35-Measurement1",
            feature_dir,
            tensor_dir,
        )

    def test_empty_feature_dir(self, config: dict, progress_path: Path, tmp_path: Path) -> None:
        """Returns False when feature dir exists but is empty."""
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        feature_dir = tmp_path / "features" / "BR00116991"
        feature_dir.mkdir(parents=True)
        tensor_dir = tmp_path / "tensors" / "BR00116991"
        assert not pipeline._check_output_complete(
            "BR00116991__2020-11-05T19_51_35-Measurement1",
            feature_dir,
            tensor_dir,
        )

    def test_completed_with_known_count(
        self, config: dict, progress_path: Path, tmp_path: Path
    ) -> None:
        """Returns True when progress says completed and file count matches."""
        plate_name = "BR00116991__2020-11-05T19_51_35-Measurement1"
        site_count = 3000  # Must exceed 80% of 3456 (=2764) threshold

        # Set up progress with a known count
        progress_data = {
            "schema_version": 1,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "config_hash": "",
            "metadata_downloaded": False,
            "plates": {
                plate_name: {
                    "status": "completed",
                    "barcode": "BR00116991",
                    "started_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "sites_extracted": site_count,
                    "error": None,
                },
            },
        }
        progress_path.write_text(json.dumps(progress_data))

        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )

        # Create feature files matching the count
        feature_dir = tmp_path / "features" / "BR00116991"
        feature_dir.mkdir(parents=True)
        for i in range(site_count):
            (feature_dir / f"site_{i:04d}.pt").touch()

        tensor_dir = tmp_path / "tensors" / "BR00116991"
        assert pipeline._check_output_complete(plate_name, feature_dir, tensor_dir)

    def test_heuristic_above_threshold(
        self, config: dict, progress_path: Path, tmp_path: Path
    ) -> None:
        """Returns True when no progress record but file count > 80% of max."""
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )

        feature_dir = tmp_path / "features" / "BR00116991"
        feature_dir.mkdir(parents=True)
        # Create 2800 files (> 80% of 3456)
        for i in range(2800):
            (feature_dir / f"site_{i:04d}.pt").touch()

        tensor_dir = tmp_path / "tensors" / "BR00116991"
        assert pipeline._check_output_complete(
            "BR00116991__2020-11-05T19_51_35-Measurement1",
            feature_dir,
            tensor_dir,
        )

    def test_heuristic_below_threshold(
        self, config: dict, progress_path: Path, tmp_path: Path
    ) -> None:
        """Returns False when file count < 80% of max and no progress record."""
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )

        feature_dir = tmp_path / "features" / "BR00116991"
        feature_dir.mkdir(parents=True)
        # Create only 100 files (way below threshold)
        for i in range(100):
            (feature_dir / f"site_{i:04d}.pt").touch()

        tensor_dir = tmp_path / "tensors" / "BR00116991"
        assert not pipeline._check_output_complete(
            "BR00116991__2020-11-05T19_51_35-Measurement1",
            feature_dir,
            tensor_dir,
        )


class TestResolvePlates:
    def test_all_plates(self, config: dict, progress_path: Path) -> None:
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        result = pipeline._resolve_plates(None)
        assert len(result) == 2

    def test_filter_by_barcode(self, config: dict, progress_path: Path) -> None:
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        result = pipeline._resolve_plates(["BR00116991"])
        assert len(result) == 1
        assert "BR00116991" in result[0]

    def test_filter_by_full_name(self, config: dict, progress_path: Path) -> None:
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        full_name = "BR00116991__2020-11-05T19_51_35-Measurement1"
        result = pipeline._resolve_plates([full_name])
        assert result == [full_name]

    def test_unknown_plate_skipped(self, config: dict, progress_path: Path) -> None:
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        result = pipeline._resolve_plates(["NONEXISTENT"])
        assert result == []


class TestRetryFailed:
    def test_retry_resets_failed_plates(self, config: dict, progress_path: Path) -> None:
        plate_name = "BR00116991__2020-11-05T19_51_35-Measurement1"
        progress_data = {
            "schema_version": 1,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "config_hash": "",
            "metadata_downloaded": False,
            "plates": {
                plate_name: {
                    "status": "failed",
                    "barcode": "BR00116991",
                    "started_at": _utcnow(),
                    "completed_at": _utcnow(),
                    "sites_extracted": 0,
                    "error": "CalledProcessError: exit code 1",
                },
            },
        }
        progress_path.write_text(json.dumps(progress_data))

        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
            retry_failed=True,
        )
        # Before reset, status should still be FAILED
        assert pipeline._progress.plates[plate_name]["status"] == PlateStatus.FAILED

        # Call _reset_failed_plates directly to verify reset logic
        reset_count = pipeline._reset_failed_plates([plate_name])
        assert reset_count == 1
        assert pipeline._progress.plates[plate_name]["status"] == PlateStatus.PENDING
        assert pipeline._progress.plates[plate_name]["error"] is None


class TestConfigHash:
    def test_hash_deterministic(self, config: dict, progress_path: Path) -> None:
        pipeline = PlateExtractionPipeline(
            config=config,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        h1 = pipeline._compute_config_hash()
        h2 = pipeline._compute_config_hash()
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_hash_changes_with_plates(self, progress_path: Path, tmp_path: Path) -> None:
        config1 = _minimal_config(tmp_path, plates=["plate_a"])
        config2 = _minimal_config(tmp_path, plates=["plate_b"])

        p1 = PlateExtractionPipeline(
            config=config1,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        p2 = PlateExtractionPipeline(
            config=config2,
            progress_path=progress_path,
            backend="awscli",
            dry_run=True,
        )
        assert p1._compute_config_hash() != p2._compute_config_hash()
