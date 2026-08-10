"""Characterize how profile columns are split into metadata and features.

Four predicates decide this across the repo and they disagree. Phase 3 replaces
them with one exact-complement pair, which is a behaviour change on the numeric
path: negcon centering runs over whatever `get_feature_columns` returns, so
moving a column between the two sets moves mAP.

These tests record today's behaviour, including the gap. When the predicates are
unified, `test_a_column_named_metadata_without_underscore_falls_through` is the
one that must be inverted, deliberately and with the export golden re-checked.
"""

import numpy as np
import pandas as pd
import pytest

from benchmark.data import get_feature_columns, get_metadata_columns
from benchmark.export_utils import negcon_center_profiles

REAL_SHAPED = pd.DataFrame(
    {
        "Metadata_Plate": ["BR00116991"],
        "Metadata_Well": ["A01"],
        "Metadata_control_type": ["negcon"],
        "feature_0000": [1.0],
        "feature_0001": [2.0],
    }
)


def test_partitions_cleanly_on_real_shaped_profiles() -> None:
    """The layout MorphoCLIP actually writes has no ambiguous column."""
    metadata = set(get_metadata_columns(REAL_SHAPED))
    features = set(get_feature_columns(REAL_SHAPED))
    assert metadata | features == set(REAL_SHAPED.columns)
    assert metadata & features == set()


def test_a_column_named_metadata_without_underscore_falls_through() -> None:
    """Today's predicates are not complements, so such a column is dropped by both.

    `get_metadata_columns` needs the trailing underscore, `get_feature_columns`
    does not, so anything matching `Metadata` but not `Metadata_` belongs to
    neither set and silently vanishes from a consensus frame.
    """
    df = REAL_SHAPED.assign(MetadataPlate=["x"])
    metadata = set(get_metadata_columns(df))
    features = set(get_feature_columns(df))

    assert "MetadataPlate" not in metadata
    assert "MetadataPlate" not in features
    assert metadata | features != set(df.columns)


def test_meta_prefixed_columns_are_treated_as_features() -> None:
    """CellCLIP's export keeps `Meta_*` columns, which this classifies as numeric.

    Recorded because it is why the `Meta_` alias cannot survive the merge: a
    non-numeric `Meta_*` column reaching negcon centering raises.
    """
    df = REAL_SHAPED.assign(Meta_Source=["source_4"])
    assert "Meta_Source" in get_feature_columns(df)

    with pytest.raises((ValueError, TypeError)):
        negcon_center_profiles(df)


def test_negcon_centering_subtracts_the_control_mean() -> None:
    profiles = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02", "A03"],
            "Metadata_control_type": ["negcon", "negcon", "trt"],
            "feature_0000": [1.0, 3.0, 10.0],
        }
    )
    centered = negcon_center_profiles(profiles)
    np.testing.assert_allclose(centered["feature_0000"], [-1.0, 1.0, 8.0])


def test_negcon_centering_falls_back_to_all_wells() -> None:
    """No negcon on the plate means centering against every well, not a crash."""
    profiles = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02"],
            "Metadata_control_type": ["trt", "trt"],
            "feature_0000": [1.0, 3.0],
        }
    )
    np.testing.assert_allclose(negcon_center_profiles(profiles)["feature_0000"], [-1.0, 1.0])
