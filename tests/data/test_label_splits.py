"""Unit tests for morphoclip.data.label_splits (pure; no torch/data needed)."""

from pathlib import Path

import pandas as pd
import pytest

from morphoclip.data.label_splits import (
    consistent_sample_split,
    create_split_keys,
    load_and_prepare_labels,
    normalize_batch,
    normalize_well,
)


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({"Well": ["A01", "B02"]}, ["A01", "B02"]),
        ({"Metadata_Well": ["A01"]}, ["A01"]),
        ({"row": ["1", "2"], "col": ["1", "12"]}, ["A01", "B12"]),
    ],
)
def test_a_well_column_is_found_or_derived_from_row_and_column(
    columns: dict[str, list[str]], expected: list[str]
) -> None:
    assert normalize_well(pd.DataFrame(columns)).tolist() == expected


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({"platecode": ["P1"], "batch": ["B1"]}, ["P1"]),
        ({"Metadata_Plate": ["P2"]}, ["P2"]),
        ({"batch": ["B3"]}, ["B3"]),
    ],
)
def test_the_batch_column_prefers_the_plate_code(
    columns: dict[str, list[str]], expected: list[str]
) -> None:
    assert normalize_batch(pd.DataFrame(columns)).tolist() == expected


def test_a_label_table_missing_its_key_columns_is_rejected() -> None:
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(ValueError, match="Well"):
        normalize_well(df)
    with pytest.raises(ValueError, match="platecode"):
        normalize_batch(df)


def test_prepared_labels_carry_a_plate_well_sample_key(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    pd.DataFrame({"platecode": ["P1", "P1"], "Well": ["A01", "A02"]}).to_csv(path, index=False)

    df = load_and_prepare_labels(path)

    assert df["UNIQUE_SAMPLE_KEY"].tolist() == ["P1-A01", "P1-A02"]
    assert (df["SAMPLE_KEY"] == df["UNIQUE_SAMPLE_KEY"]).all()
    assert df["treatment"].tolist() == ["A01", "A02"]
    assert (df["prompt"] == "").all()


def test_a_group_splits_on_sorted_treatment_ids_not_row_order() -> None:
    group = pd.DataFrame(
        {"treatment": ["d", "c", "b", "a"], "UNIQUE_SAMPLE_KEY": ["k4", "k3", "k2", "k1"]}
    )

    train, test = consistent_sample_split(group, train_ratio=0.5)

    assert sorted(train["treatment"].tolist()) == ["a", "b"]
    assert sorted(test["treatment"].tolist()) == ["c", "d"]


def test_split_keys_partition_every_group_without_overlap() -> None:
    df = pd.DataFrame(
        {
            "batch": ["B1", "B1", "B2", "B2"],
            "treatment": ["a", "b", "c", "d"],
            "UNIQUE_SAMPLE_KEY": ["B1-a", "B1-b", "B2-c", "B2-d"],
        }
    )

    train_keys, test_keys = create_split_keys(df, ["batch"], train_ratio=0.5)

    assert sorted(train_keys) == ["B1-a", "B2-c"]
    assert sorted(test_keys) == ["B1-b", "B2-d"]
