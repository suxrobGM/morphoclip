"""Tests for the plate-at-a-time CellCLIP pipeline helpers."""

from pathlib import Path

import pytest
import torch

from cellclip.benchmark.pipeline import (
    PlatePaths,
    cached_feature_width,
    discover_downloaded_plate_dirs,
    resolve_plate_paths,
    run_plate_pipeline,
)


def _touch_site_images(image_dir: Path, *, sites: list[str]) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for site in sites:
        for channel in range(1, 6):
            (image_dir / f"{site}p01-ch{channel}sk1fk1fl1.png").touch()


def test_discover_downloaded_plate_dirs_uses_batch_layout(tmp_path: Path) -> None:
    batch_root = tmp_path / "raw_compressed" / "2020_11_04_CPJUMP1"
    image_dir = batch_root / "BR00116991__2020-11-05T19_51_35-Measurement1" / "Images"
    _touch_site_images(image_dir, sites=["r01c01f01"])

    discovered = discover_downloaded_plate_dirs(tmp_path / "raw_compressed", "2020_11_04_CPJUMP1")
    assert discovered == {"BR00116991": image_dir}


def test_resolve_plate_paths_reports_missing_downloads(tmp_path: Path) -> None:
    compressed_root = tmp_path / "raw_compressed"
    image_dir = (
        compressed_root
        / "2020_11_04_CPJUMP1"
        / "BR00116991__2020-11-05T19_51_35-Measurement1"
        / "Images"
    )
    _touch_site_images(image_dir, sites=["r01c01f01"])

    resolved, missing = resolve_plate_paths(
        barcodes=["BR00116991", "BR00116992"],
        compressed_root=compressed_root,
        features_root=tmp_path / "features",
        tensors_root=tmp_path / "tensors",
        output_profiles_root=tmp_path / "profiles",
        batch="2020_11_04_CPJUMP1",
    )

    assert [item.barcode for item in resolved] == ["BR00116991"]
    assert missing == ["BR00116992"]


def test_run_plate_pipeline_reuses_and_cleans_complete_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "compressed" / "2020_11_04_CPJUMP1" / "BR00116991__ts" / "Images"
    _touch_site_images(image_dir, sites=["r01c01f01"])

    feature_dir = tmp_path / "features" / "BR00116991"
    feature_dir.mkdir(parents=True)
    torch.save(torch.zeros(5, 1536), feature_dir / "r01c01f01.pt")

    output_path = (
        tmp_path
        / "profiles"
        / "2020_11_04_CPJUMP1"
        / "BR00116991"
        / "BR00116991_normalized_feature_select_negcon_batch.csv.gz"
    )

    def fail_extract(**kwargs) -> None:
        raise AssertionError("extraction should not run when the cache is complete")

    def fake_export(**kwargs) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("ok", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "cellclip.benchmark.pipeline.extract_plate_features_with_model", fail_extract
    )
    monkeypatch.setattr("cellclip.benchmark.pipeline.export_plate", fake_export)

    result = run_plate_pipeline(
        plate=PlatePaths(
            barcode="BR00116991",
            image_dir=image_dir,
            feature_dir=feature_dir,
            tensor_dir=tmp_path / "tensors" / "BR00116991",
            output_path=output_path,
        ),
        dino_model=object(),
        dino_processor=object(),
        dino_device="cpu",
        extraction_batch_size=8,
        save_tensors=False,
        cellclip_model=object(),
        cellclip_device="cpu",
        source_profiles_root=tmp_path / "source_profiles",
        batch="2020_11_04_CPJUMP1",
        site_batch_size=4,
        input_dim=1536,
        force_export=False,
        keep_features=False,
        keep_tensors=False,
        prune_empty_dirs=True,
    )

    assert result.status == "reused_existing_features"
    assert result.features_deleted == 1
    assert output_path.exists()
    assert not feature_dir.exists()


def test_run_plate_pipeline_keeps_features_when_export_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "compressed" / "2020_11_04_CPJUMP1" / "BR00116992__ts" / "Images"
    _touch_site_images(image_dir, sites=["r01c01f01", "r01c01f02"])

    feature_dir = tmp_path / "features" / "BR00116992"

    def fake_extract(**kwargs) -> None:
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "r01c01f01.pt").touch()
        (output_dir / "r01c01f02.pt").touch()

    def fail_export(**kwargs) -> Path:
        raise RuntimeError("export failed")

    monkeypatch.setattr(
        "cellclip.benchmark.pipeline.extract_plate_features_with_model", fake_extract
    )
    monkeypatch.setattr("cellclip.benchmark.pipeline.export_plate", fail_export)

    with pytest.raises(RuntimeError, match="export failed"):
        run_plate_pipeline(
            plate=PlatePaths(
                barcode="BR00116992",
                image_dir=image_dir,
                feature_dir=feature_dir,
                tensor_dir=tmp_path / "tensors" / "BR00116992",
                output_path=tmp_path
                / "profiles"
                / "2020_11_04_CPJUMP1"
                / "BR00116992"
                / "BR00116992_normalized_feature_select_negcon_batch.csv.gz",
            ),
            dino_model=object(),
            dino_processor=object(),
            dino_device="cpu",
            extraction_batch_size=8,
            save_tensors=False,
            cellclip_model=object(),
            cellclip_device="cpu",
            source_profiles_root=tmp_path / "source_profiles",
            batch="2020_11_04_CPJUMP1",
            site_batch_size=4,
            input_dim=1536,
            force_export=False,
            keep_features=False,
            keep_tensors=False,
            prune_empty_dirs=True,
        )

    assert (feature_dir / "r01c01f01.pt").exists()
    assert (feature_dir / "r01c01f02.pt").exists()


def test_cached_feature_width_reads_saved_tensor_width(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features" / "BR00116993"
    feature_dir.mkdir(parents=True)
    torch.save(torch.zeros(5, 1536), feature_dir / "r01c01f01.pt")

    assert cached_feature_width(feature_dir) == 1536


def test_run_plate_pipeline_reextracts_when_cached_width_is_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "compressed" / "2020_11_04_CPJUMP1" / "BR00116994__ts" / "Images"
    _touch_site_images(image_dir, sites=["r01c01f01"])

    feature_dir = tmp_path / "features" / "BR00116994"
    feature_dir.mkdir(parents=True)
    torch.save(torch.zeros(5, 1024), feature_dir / "r01c01f01.pt")

    output_path = (
        tmp_path
        / "profiles"
        / "2020_11_04_CPJUMP1"
        / "BR00116994"
        / "BR00116994_normalized_feature_select_negcon_batch.csv.gz"
    )

    def fake_extract(**kwargs) -> None:
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.zeros(5, 1536), output_dir / "r01c01f01.pt")

    def fake_export(**kwargs) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("ok", encoding="utf-8")
        return output_path

    monkeypatch.setattr(
        "cellclip.benchmark.pipeline.extract_plate_features_with_model", fake_extract
    )
    monkeypatch.setattr("cellclip.benchmark.pipeline.export_plate", fake_export)

    result = run_plate_pipeline(
        plate=PlatePaths(
            barcode="BR00116994",
            image_dir=image_dir,
            feature_dir=feature_dir,
            tensor_dir=tmp_path / "tensors" / "BR00116994",
            output_path=output_path,
        ),
        dino_model=object(),
        dino_processor=object(),
        dino_device="cpu",
        extraction_batch_size=8,
        save_tensors=False,
        cellclip_model=object(),
        cellclip_device="cpu",
        source_profiles_root=tmp_path / "source_profiles",
        batch="2020_11_04_CPJUMP1",
        site_batch_size=4,
        input_dim=1536,
        force_export=False,
        keep_features=True,
        keep_tensors=False,
        prune_empty_dirs=False,
    )

    assert result.status == "exported"
    assert "enforce width 1536" in result.message
    assert cached_feature_width(feature_dir) == 1536
