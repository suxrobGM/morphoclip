"""Tests for benchmark split-manifest data helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark.data import (
    filter_experiment_metadata_to_split_subset,
    filter_profiles_to_split_subset,
    load_split_manifest,
)


def test_load_split_manifest_accepts_val_alias(tmp_path: Path) -> None:
    manifest_path = tmp_path / "split_manifest.csv"
    pd.DataFrame(
        {
            "subset": ["train", "validate", "test"],
            "Metadata_Plate": ["BR00117003", "BR00116991", "BR00117017"],
            "Metadata_Well": ["A01", "A01", "A01"],
        }
    ).to_csv(manifest_path, index=False)

    validate_rows = load_split_manifest(manifest_path, "val")

    assert validate_rows[["Metadata_Plate", "Metadata_Well"]].values.tolist() == [
        ["BR00116991", "A01"]
    ]


def test_filter_profiles_to_split_subset_keeps_all_sites_for_selected_well() -> None:
    profiles = pd.DataFrame(
        {
            "Metadata_Plate": ["BR00117017", "BR00117017", "BR00117017", "BR00117003"],
            "Metadata_Well": ["A01", "A01", "A03", "A01"],
            "Metadata_Site": ["f01", "f02", "f01", "f01"],
            "feature_0000": [1.0, 1.1, 2.0, 3.0],
        }
    )
    split_manifest = pd.DataFrame(
        {
            "subset": ["test"],
            "Metadata_Plate": ["BR00117017"],
            "Metadata_Well": ["A01"],
        }
    )

    filtered = filter_profiles_to_split_subset(profiles, split_manifest)

    assert filtered["Metadata_Well"].tolist() == ["A01", "A01"]
    assert filtered["Metadata_Site"].tolist() == ["f01", "f02"]


def test_filter_profiles_to_split_subset_can_keep_negcons_on_selected_plates() -> None:
    profiles = pd.DataFrame(
        {
            "Metadata_Plate": ["BR00117017", "BR00117017", "BR00117017", "BR00117003"],
            "Metadata_Well": ["A01", "A02", "A03", "A01"],
            "Metadata_control_type": ["trt", "negcon", "trt", "negcon"],
            "Metadata_Site": ["f01", "f01", "f01", "f01"],
            "feature_0000": [1.0, 1.1, 2.0, 3.0],
        }
    )
    split_manifest = pd.DataFrame(
        {
            "subset": ["test"],
            "Metadata_Plate": ["BR00117017"],
            "Metadata_Well": ["A01"],
        }
    )

    filtered = filter_profiles_to_split_subset(
        profiles,
        split_manifest,
        keep_negcon_on_selected_plates=True,
    )

    assert filtered[["Metadata_Well", "Metadata_control_type"]].values.tolist() == [
        ["A01", "trt"],
        ["A02", "negcon"],
    ]


def test_filter_experiment_metadata_to_split_subset_filters_to_selected_plates() -> None:
    experiment_df = pd.DataFrame(
        {
            "Assay_Plate_Barcode": ["BR00117017", "BR00117003", "BR00117020"],
            "Perturbation": ["compound", "crispr", "orf"],
        }
    )
    split_manifest = pd.DataFrame(
        {
            "subset": ["test", "test"],
            "Metadata_Plate": ["BR00117017", "BR00117017"],
            "Metadata_Well": ["A01", "A03"],
        }
    )

    filtered = filter_experiment_metadata_to_split_subset(experiment_df, split_manifest)

    assert filtered["Assay_Plate_Barcode"].tolist() == ["BR00117017"]
