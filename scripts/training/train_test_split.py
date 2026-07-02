#!/usr/bin/env python
"""Create train/test splits from `labels.csv`.

Conventions for this project:
- `batch` = plate number (`platecode` / `Metadata_Plate`)
- `UNIQUE_SAMPLE_KEY` = `{batch}-{Well}`
- `treatment` = `Well`
- `prompt` = empty string for now (placeholder)
"""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/test split from labels.csv (plate+well key)."
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("output/labels.csv"),
        help="Path to labels.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/train_test_split"),
        help="Directory for split output CSVs",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Train ratio used within each group (deterministic, sorted by treatment)",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=["batch"],
        help="Grouping columns used before per-group well split",
    )
    return parser.parse_args()


def _normalize_well(df: pd.DataFrame) -> pd.Series:
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


def _normalize_batch(df: pd.DataFrame) -> pd.Series:
    if "platecode" in df.columns:
        return df["platecode"].astype(str)
    if "Metadata_Plate" in df.columns:
        return df["Metadata_Plate"].astype(str)
    if "batch" in df.columns:
        return df["batch"].astype(str)
    raise ValueError("Label file must contain `platecode` or `Metadata_Plate` or `batch`.")


def load_and_prepare_labels(label_file: Path) -> pd.DataFrame:
    full_df = pd.read_csv(label_file)
    full_df["batch"] = _normalize_batch(full_df)
    full_df["Well"] = _normalize_well(full_df)
    full_df["UNIQUE_SAMPLE_KEY"] = full_df["batch"] + "-" + full_df["Well"]
    full_df["SAMPLE_KEY"] = full_df["UNIQUE_SAMPLE_KEY"]
    full_df["treatment"] = full_df["Well"]
    full_df["prompt"] = ""  # TODO: add prompt
    return full_df


def consistent_sample_split(
    group: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = group["treatment"].dropna().unique().tolist()
    ids = sorted(ids)

    train_size = int(train_ratio * len(ids))
    train_ids = set(ids[:train_size])
    test_ids = set(ids[train_size:])

    train_group = group[group["treatment"].isin(train_ids)]
    test_group = group[group["treatment"].isin(test_ids)]
    return train_group, test_group


def create_split_keys(
    merged_df: pd.DataFrame, group_columns: list[str], train_ratio: float
) -> tuple[list[str], list[str]]:
    train_keys: list[str] = []
    test_keys: list[str] = []

    for _, group in merged_df.groupby(group_columns, dropna=False):
        train_group, test_group = consistent_sample_split(group, train_ratio)
        train_keys.extend(train_group["UNIQUE_SAMPLE_KEY"].tolist())
        test_keys.extend(test_group["UNIQUE_SAMPLE_KEY"].tolist())

    return train_keys, test_keys


def materialize_split_labels(
    full_label_df: pd.DataFrame, train_keys: list[str], test_keys: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_label = full_label_df[full_label_df["UNIQUE_SAMPLE_KEY"].isin(train_keys)].copy()
    test_label = full_label_df[full_label_df["UNIQUE_SAMPLE_KEY"].isin(test_keys)].copy()
    return train_label, test_label


def main() -> None:
    args = parse_args()

    full_label_df = load_and_prepare_labels(args.label_file)
    key_df = full_label_df.drop_duplicates(subset=["UNIQUE_SAMPLE_KEY"]).reset_index(drop=True)

    required_columns = set(args.group_columns + ["treatment", "UNIQUE_SAMPLE_KEY"])
    missing = required_columns - set(key_df.columns)
    if missing:
        raise ValueError(f"Label table missing required columns: {sorted(missing)}")

    train_keys, test_keys = create_split_keys(
        merged_df=key_df,
        group_columns=args.group_columns,
        train_ratio=args.train_ratio,
    )

    if set(train_keys) & set(test_keys):
        raise ValueError("Train and test keys overlap.")

    train_label, test_label = materialize_split_labels(full_label_df, train_keys, test_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "jumpcp_training_label.csv"
    test_out = args.output_dir / "jumpcp_testing_label.csv"

    train_label.to_csv(train_out, index=False)
    test_label.to_csv(test_out, index=False)

    print(f"Train labels: {len(train_label)}, Test labels: {len(test_label)}")
    print(f"Saved: {train_out}")
    print(f"Saved: {test_out}")


if __name__ == "__main__":
    main()
