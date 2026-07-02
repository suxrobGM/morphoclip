"""Unit tests for morphoclip.data.label_splits (pure; no torch/data needed)."""

import pandas as pd
import pytest

from morphoclip.data.label_splits import (
    consistent_sample_split,
    create_split_keys,
    load_and_prepare_labels,
    normalize_batch,
    normalize_well,
)


class TestNormalizeWell:
    def test_well_column(self):
        df = pd.DataFrame({"Well": ["A01", "B02"]})
        assert normalize_well(df).tolist() == ["A01", "B02"]

    def test_metadata_well_column(self):
        df = pd.DataFrame({"Metadata_Well": ["A01"]})
        assert normalize_well(df).tolist() == ["A01"]

    def test_row_col_derivation(self):
        df = pd.DataFrame({"row": ["1", "2"], "col": ["1", "12"]})
        assert normalize_well(df).tolist() == ["A01", "B12"]

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="Well"):
            normalize_well(pd.DataFrame({"foo": [1]}))


class TestNormalizeBatch:
    def test_platecode_preferred(self):
        df = pd.DataFrame({"platecode": ["P1"], "batch": ["B1"]})
        assert normalize_batch(df).tolist() == ["P1"]

    def test_metadata_plate(self):
        df = pd.DataFrame({"Metadata_Plate": ["P2"]})
        assert normalize_batch(df).tolist() == ["P2"]

    def test_batch_fallback(self):
        df = pd.DataFrame({"batch": ["B3"]})
        assert normalize_batch(df).tolist() == ["B3"]

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="platecode"):
            normalize_batch(pd.DataFrame({"foo": [1]}))


class TestLoadAndPrepareLabels:
    def test_builds_keys(self, tmp_path):
        path = tmp_path / "labels.csv"
        pd.DataFrame({"platecode": ["P1", "P1"], "Well": ["A01", "A02"]}).to_csv(path, index=False)
        df = load_and_prepare_labels(path)
        assert df["UNIQUE_SAMPLE_KEY"].tolist() == ["P1-A01", "P1-A02"]
        assert (df["SAMPLE_KEY"] == df["UNIQUE_SAMPLE_KEY"]).all()
        assert df["treatment"].tolist() == ["A01", "A02"]
        assert (df["prompt"] == "").all()


class TestConsistentSampleSplit:
    def test_deterministic_by_sorted_treatment(self):
        group = pd.DataFrame(
            {"treatment": ["d", "c", "b", "a"], "UNIQUE_SAMPLE_KEY": ["k4", "k3", "k2", "k1"]}
        )
        train, test = consistent_sample_split(group, train_ratio=0.5)
        # sorted ids = [a, b, c, d]; train_size = 2 -> {a, b}
        assert sorted(train["treatment"].tolist()) == ["a", "b"]
        assert sorted(test["treatment"].tolist()) == ["c", "d"]


class TestCreateSplitKeys:
    def test_per_group_split_no_overlap(self):
        df = pd.DataFrame(
            {
                "batch": ["B1", "B1", "B2", "B2"],
                "treatment": ["a", "b", "c", "d"],
                "UNIQUE_SAMPLE_KEY": ["B1-a", "B1-b", "B2-c", "B2-d"],
            }
        )
        train_keys, test_keys = create_split_keys(df, ["batch"], train_ratio=0.5)
        assert not set(train_keys) & set(test_keys)
        assert set(train_keys) | set(test_keys) == {"B1-a", "B1-b", "B2-c", "B2-d"}
        # each group of 2 splits 1/1
        assert len(train_keys) == 2 and len(test_keys) == 2
