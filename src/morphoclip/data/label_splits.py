"""Deterministic train/test label splitting from a labels.csv.

Used by the ``morphoclip split`` command to turn a plate+well ``labels.csv`` into
train/test label tables. This is distinct from :mod:`morphoclip.data.splits`,
which splits a :class:`MorphoCLIPDataset` into torch ``Subset`` objects for
training; here we produce deterministic per-group sample keys from a CSV.
"""

from pathlib import Path

import pandas as pd


def normalize_well(df: pd.DataFrame) -> pd.Series:
    """Derive a canonical ``Well`` series (e.g. ``A01``) from label columns."""
    if "Well" in df.columns:
        return df["Well"].astype(str)
    if "Metadata_Well" in df.columns:
        return df["Metadata_Well"].astype(str)
    if "row" in df.columns and "col" in df.columns:
        row_num = df["row"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
        col_num = df["col"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
        row_letter = row_num.apply(lambda x: chr(ord("A") + x - 1))
        return row_letter + col_num.map(lambda x: f"{x:02d}")
    raise ValueError("Label file must contain `Well`, `Metadata_Well`, or (`row`,`col`).")


def normalize_batch(df: pd.DataFrame) -> pd.Series:
    """Derive a canonical ``batch`` series from label columns."""
    if "platecode" in df.columns:
        return df["platecode"].astype(str)
    if "Metadata_Plate" in df.columns:
        return df["Metadata_Plate"].astype(str)
    if "batch" in df.columns:
        return df["batch"].astype(str)
    raise ValueError("Label file must contain `platecode` or `Metadata_Plate` or `batch`.")


def load_and_prepare_labels(label_file: Path) -> pd.DataFrame:
    """Load a labels.csv and add the derived keys the splitter needs."""
    full_df = pd.read_csv(label_file)
    full_df["batch"] = normalize_batch(full_df)
    full_df["Well"] = normalize_well(full_df)
    full_df["UNIQUE_SAMPLE_KEY"] = full_df["batch"] + "-" + full_df["Well"]
    full_df["SAMPLE_KEY"] = full_df["UNIQUE_SAMPLE_KEY"]
    full_df["treatment"] = full_df["Well"]
    full_df["prompt"] = ""
    return full_df


def consistent_sample_split(
    group: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one group's rows into train/test by sorted, deduplicated treatment id."""
    ids = sorted(group["treatment"].dropna().unique().tolist())
    train_size = int(train_ratio * len(ids))
    train_ids = set(ids[:train_size])
    test_ids = set(ids[train_size:])
    return group[group["treatment"].isin(train_ids)], group[group["treatment"].isin(test_ids)]


def create_split_keys(
    merged_df: pd.DataFrame, group_columns: list[str], train_ratio: float
) -> tuple[list[str], list[str]]:
    """Build train/test ``UNIQUE_SAMPLE_KEY`` lists via per-group treatment splits."""
    train_keys: list[str] = []
    test_keys: list[str] = []
    for _, group in merged_df.groupby(group_columns, dropna=False):
        train_group, test_group = consistent_sample_split(group, train_ratio)
        train_keys.extend(train_group["UNIQUE_SAMPLE_KEY"].tolist())
        test_keys.extend(test_group["UNIQUE_SAMPLE_KEY"].tolist())
    return train_keys, test_keys
