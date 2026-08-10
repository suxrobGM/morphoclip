"""Tests for MorphoCLIP benchmark-layout profile export helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from morphoclip.benchmark.export import build_plate_profile, load_reference_metadata


def _metadata_df(wells: list[str], **extra_cols: list) -> pd.DataFrame:
    data = {
        "Metadata_Plate": ["BR00116991"] * len(wells),
        "Metadata_Well": wells,
    }
    data.update(extra_cols)
    return pd.DataFrame(data)


class TestLoadReferenceMetadata:
    def test_filters_to_requested_plate(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metadata.csv"
        pd.DataFrame(
            {
                "Metadata_Plate": ["BR00116991", "BR00116991", "BR00116992"],
                "Metadata_Well": ["A01", "A02", "A01"],
                "Metadata_broad_sample": ["x1", "x2", "x3"],
            }
        ).to_csv(csv_path, index=False)

        result = load_reference_metadata(csv_path, "BR00116991")

        assert sorted(result["Metadata_Well"]) == ["A01", "A02"]
        assert (result["Metadata_Plate"] == "BR00116991").all()

    def test_raises_on_duplicate_well(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metadata.csv"
        pd.DataFrame(
            {
                "Metadata_Plate": ["BR00116991", "BR00116991"],
                "Metadata_Well": ["A01", "A01"],
            }
        ).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="duplicate"):
            load_reference_metadata(csv_path, "BR00116991")

    def test_raises_when_plate_not_found(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metadata.csv"
        pd.DataFrame({"Metadata_Plate": ["BR00116991"], "Metadata_Well": ["A01"]}).to_csv(
            csv_path, index=False
        )

        with pytest.raises(ValueError, match="No reference metadata"):
            load_reference_metadata(csv_path, "BR00999999")


class TestBuildPlateProfile:
    def test_aligns_wells_and_passes_through_metadata(self) -> None:
        metadata_df = _metadata_df(
            ["A01", "A02", "A03"],
            Metadata_broad_sample=["s1", "s2", "s3"],
            Metadata_control_type=["", "", "negcon"],
        )
        rng = np.random.default_rng(0)
        embeddings = rng.standard_normal((3, 4)).astype(np.float32)
        wells = ["A03", "A01", "A02"]  # deliberately out of metadata order

        profile = build_plate_profile(embeddings, wells, metadata_df)

        assert list(profile["Metadata_Well"]) == ["A01", "A02", "A03"]
        assert list(profile["Metadata_broad_sample"]) == ["s1", "s2", "s3"]
        # embedding for A01 (index 1 in `wells`) lands on the A01 row.
        a01_row = profile.loc[profile["Metadata_Well"] == "A01", "feature_0000":"feature_0003"]
        np.testing.assert_allclose(a01_row.to_numpy()[0], embeddings[1])

    def test_drops_embedding_well_without_metadata(self, caplog: pytest.LogCaptureFixture) -> None:
        metadata_df = _metadata_df(["A01", "A02"])
        embeddings = np.zeros((3, 2), dtype=np.float32)
        wells = ["A01", "A02", "Z99"]

        with caplog.at_level("WARNING"):
            profile = build_plate_profile(embeddings, wells, metadata_df)

        assert sorted(profile["Metadata_Well"]) == ["A01", "A02"]
        assert any("no reference metadata" in msg for msg in caplog.messages)

    def test_reports_metadata_well_without_embedding(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        metadata_df = _metadata_df(["A01", "A02", "A03"])
        embeddings = np.zeros((2, 2), dtype=np.float32)
        wells = ["A01", "A02"]

        with caplog.at_level("WARNING"):
            profile = build_plate_profile(embeddings, wells, metadata_df)

        assert sorted(profile["Metadata_Well"]) == ["A01", "A02"]
        assert any("no encoded embedding" in msg for msg in caplog.messages)

    def test_raises_on_duplicate_embedding_wells(self) -> None:
        metadata_df = _metadata_df(["A01", "A02"])
        embeddings = np.zeros((2, 2), dtype=np.float32)
        wells = ["A01", "A01"]

        with pytest.raises(ValueError, match="Duplicate wells"):
            build_plate_profile(embeddings, wells, metadata_df)

    def test_accepts_torch_tensor_embeddings(self) -> None:
        metadata_df = _metadata_df(["A01", "A02"])
        embeddings = torch.randn(2, 5)
        wells = ["A01", "A02"]

        profile = build_plate_profile(embeddings, wells, metadata_df)

        assert profile.shape == (2, 2 + 5)

    def test_feature_column_naming_and_count(self) -> None:
        metadata_df = _metadata_df(["A01"])
        embeddings = np.zeros((1, 512), dtype=np.float32)

        profile = build_plate_profile(embeddings, ["A01"], metadata_df)

        feature_cols = [c for c in profile.columns if c.startswith("feature_")]
        assert len(feature_cols) == 512
        assert feature_cols[0] == "feature_0000"
        assert feature_cols[-1] == "feature_0511"
