"""Tests for the on-disk profile contract.

Four predicates used to decide which columns are features. They disagreed, and
the split is on the numeric path: negcon centering runs over whatever counts as
a feature, so moving one column changes mAP. `metadata_columns` and
`feature_columns` are now exact complements over `Metadata_` and `Meta_`.

The change is safe on real data, not just in principle: all 80 profile CSVs
under `data/profiles_*` are 29 `Metadata_*` + 512 `feature_*` with zero `Meta_*`
and zero ambiguous columns, so no column moves sides.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.profiles import (
    PROFILE_SUFFIX,
    feature_columns,
    feature_frame,
    metadata_columns,
    metadata_frame,
    negcon_center_profiles,
    numbered_feature_columns,
    profile_path,
    write_profile,
)

REAL_SHAPED = pd.DataFrame(
    {
        "Metadata_Plate": ["BR00116991"],
        "Metadata_Well": ["A01"],
        "Metadata_control_type": ["negcon"],
        "feature_0000": [1.0],
        "feature_0001": [2.0],
    }
)


class TestColumnSplit:
    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"Meta_Source": ["source_4"]},
            {"MetadataPlate": ["x"]},
            {"Metadata": ["x"]},
            {"metadata_plate": ["x"]},
        ],
    )
    def test_every_column_lands_in_exactly_one_set(self, extra: dict) -> None:
        df = REAL_SHAPED.assign(**extra)
        metadata = set(metadata_columns(df))
        features = set(feature_columns(df))
        assert metadata | features == set(df.columns)
        assert metadata & features == set()

    def test_the_meta_alias_counts_as_metadata(self) -> None:
        """CellCLIP source profiles use it. Treating it as numeric used to crash centering."""
        df = REAL_SHAPED.assign(Meta_Source=["source_4"])
        assert "Meta_Source" in metadata_columns(df)
        assert "Meta_Source" not in feature_columns(df)

    def test_a_metadata_column_without_the_underscore_is_a_feature(self) -> None:
        """Inverted deliberately: it used to fall through both predicates and vanish."""
        df = REAL_SHAPED.assign(MetadataPlate=["x"])
        assert "MetadataPlate" in feature_columns(df)

    def test_the_frames_keep_column_order(self) -> None:
        assert list(metadata_frame(REAL_SHAPED).columns) == metadata_columns(REAL_SHAPED)
        assert list(feature_frame(REAL_SHAPED).columns) == feature_columns(REAL_SHAPED)

    def test_numbered_columns_are_zero_padded_to_four_digits(self) -> None:
        names = numbered_feature_columns(3)
        assert names == ["feature_0000", "feature_0001", "feature_0002"]
        assert numbered_feature_columns(10000)[-1] == "feature_9999"

    def test_freshly_numbered_columns_read_back_as_features(self) -> None:
        df = pd.DataFrame(np.zeros((1, 3)), columns=numbered_feature_columns(3))
        assert feature_columns(df) == numbered_feature_columns(3)


class TestNegconCentering:
    def test_the_control_mean_is_subtracted_from_every_well(self) -> None:
        profiles = pd.DataFrame(
            {
                "Metadata_Well": ["A01", "A02", "A03"],
                "Metadata_control_type": ["negcon", "negcon", "trt"],
                "feature_0000": [1.0, 3.0, 10.0],
                "feature_0001": [2.0, 4.0, 10.0],
            }
        )
        centered = negcon_center_profiles(profiles)
        np.testing.assert_allclose(centered["feature_0000"], [-1.0, 1.0, 8.0])
        np.testing.assert_allclose(centered["feature_0001"], [-1.0, 1.0, 7.0])

    def test_a_plate_with_no_controls_centers_on_all_wells(self) -> None:
        profiles = pd.DataFrame(
            {
                "Metadata_Well": ["A01", "A02"],
                "Metadata_control_type": ["trt", "trt"],
                "feature_0000": [1.0, 3.0],
            }
        )
        np.testing.assert_allclose(negcon_center_profiles(profiles)["feature_0000"], [-1.0, 1.0])

    def test_a_missing_control_column_centers_on_all_wells(self) -> None:
        profiles = pd.DataFrame({"Metadata_Well": ["A01", "A02"], "feature_0000": [1.0, 3.0]})
        np.testing.assert_allclose(negcon_center_profiles(profiles)["feature_0000"], [-1.0, 1.0])

    def test_metadata_columns_are_left_alone(self) -> None:
        centered = negcon_center_profiles(REAL_SHAPED)
        pd.testing.assert_frame_equal(metadata_frame(centered), metadata_frame(REAL_SHAPED))

    def test_a_frame_with_no_features_is_returned_unchanged(self) -> None:
        metadata_only = REAL_SHAPED[metadata_columns(REAL_SHAPED)]
        assert negcon_center_profiles(metadata_only) is metadata_only

    def test_a_meta_prefixed_column_no_longer_breaks_centering(self) -> None:
        """This raised before the merge, because `Meta_Source` counted as numeric."""
        centered = negcon_center_profiles(REAL_SHAPED.assign(Meta_Source=["source_4"]))
        assert centered["Meta_Source"].tolist() == ["source_4"]

    def test_the_input_frame_is_not_mutated(self) -> None:
        profiles = REAL_SHAPED.copy()
        negcon_center_profiles(profiles)
        pd.testing.assert_frame_equal(profiles, REAL_SHAPED)


class TestProfilePath:
    def test_matches_the_layout_profileloader_globs_for(self, tmp_path: Path) -> None:
        path = profile_path(tmp_path, "2020_11_04_CPJUMP1", "BR00116991")
        assert path == (
            tmp_path / "2020_11_04_CPJUMP1" / "BR00116991" / f"BR00116991_{PROFILE_SUFFIX}"
        )

    def test_write_profile_creates_missing_directories(self, tmp_path: Path) -> None:
        path = write_profile(REAL_SHAPED, profile_path(tmp_path, "BATCH", "PLATE"))
        assert path.exists()
        pd.testing.assert_frame_equal(pd.read_csv(path), REAL_SHAPED)
