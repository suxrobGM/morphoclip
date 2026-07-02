#!/usr/bin/env python
"""Generate label rows from CPJUMP image folders.

Behavior:
- Take a batch folder.
- For each plate folder under batch, enter `<plate_folder>/Images`.
- Parse files named like: `rXXcXXfXXp01-chXXsk1fk1fl1.tiff`.
- Build `Well` from row/col (`r01,c01 -> A01`).
- Group by `(platecode, row, col, site)` and count unique channels.
- For each plate, load profile file:
  `{platecode}_normalized_feature_select_negcon_batch.csv.gz`.
- Keep metadata columns (prefix `Meta_` or `Metadata_`) and join on
  `Well == Metadata_Well`.
- If channel count >= 5, write to `output/labels.csv`.
- If channel count < 5, write to `output/incomplete_channel_wells.csv`
  with an extra `channel_count` column.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

FILE_RE = re.compile(
    r"^r(?P<row>\d{2})c(?P<col>\d{2})f(?P<site>\d{2})p(?P<plane>\d{2})-ch(?P<channel>\d{1,2})sk1fk1fl1\.tiff$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labels from batch/plate/Images folders.")
    parser.add_argument(
        "--batch-folder",
        type=Path,
        default=Path("data/raw/2020_11_04_CPJUMP1"),
        help="Batch folder that contains plate directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory for generated CSV files.",
    )
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=Path("data/profiles"),
        help="Root profiles directory containing per-batch plate folders.",
    )
    return parser.parse_args()


def row_to_letter(row_token: str) -> str:
    row_num = int(row_token[1:])
    if row_num < 1 or row_num > 26:
        raise ValueError(f"Unsupported row index for Well conversion: {row_token}")
    return chr(ord("A") + row_num - 1)


def to_well(row_token: str, col_token: str) -> str:
    row_letter = row_to_letter(row_token)
    col_num = int(col_token[1:])
    return f"{row_letter}{col_num:02d}"


def parse_plate_images(plate_folder: Path, plate_code: str) -> pd.DataFrame:
    """Parse valid image filenames under one plate's `Images` folder."""
    images_dir = plate_folder / "Images"
    if not images_dir.exists() or not images_dir.is_dir():
        print(f"Skipping {plate_folder.name}: missing Images/")
        return pd.DataFrame(columns=["platecode", "row", "col", "site", "channel", "Well"])

    records: list[dict[str, str]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue

        match = FILE_RE.match(image_path.name)
        if match is None:
            continue

        channel = int(match.group("channel"))

        row_token = f"r{match.group('row')}"
        col_token = f"c{match.group('col')}"

        records.append(
            {
                "platecode": plate_code,
                "row": row_token,
                "col": col_token,
                "site": f"f{match.group('site')}",
                "channel": f"ch{channel:02d}",
                "Well": to_well(row_token, col_token),
            }
        )
    return pd.DataFrame.from_records(records)


def load_plate_profile_metadata(
    profiles_root: Path, batch_name: str, plate_code: str
) -> pd.DataFrame:
    profile_path = (
        profiles_root
        / batch_name
        / plate_code
        / f"{plate_code}_normalized_feature_select_negcon_batch.csv.gz"
    )
    if not profile_path.exists():
        print(f"Warning: profile not found for plate {plate_code}: {profile_path}")
        return pd.DataFrame(columns=["Metadata_Well"])

    header = pd.read_csv(profile_path, nrows=0)
    meta_cols = [c for c in header.columns if c.startswith("Meta_") or c.startswith("Metadata_")]

    if "Metadata_Well" not in meta_cols and "Metadata_Well" in header.columns:
        meta_cols.append("Metadata_Well")

    if not meta_cols:
        print(f"Warning: no metadata columns found in profile for {plate_code}")
        return pd.DataFrame(columns=["Metadata_Well"])

    meta_df = pd.read_csv(profile_path, usecols=meta_cols)
    if "Metadata_Well" in meta_df.columns:
        meta_df["Metadata_Well"] = meta_df["Metadata_Well"].astype(str)
        meta_df = meta_df.drop_duplicates(subset=["Metadata_Well"]).reset_index(drop=True)
    return meta_df


def main() -> None:
    args = parse_args()
    batch_folder = args.batch_folder.resolve()
    batch_name = batch_folder.name

    if not batch_folder.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_folder}")

    plate_folders = sorted(path for path in batch_folder.iterdir() if path.is_dir())
    if not plate_folders:
        print(f"No plate folders found under: {batch_folder}")
        return

    per_plate_grouped: list[pd.DataFrame] = []
    for plate_folder in plate_folders:
        plate_code = plate_folder.name.split("_")[0]
        print(plate_code)
        plate_df = parse_plate_images(plate_folder, plate_code)
        if plate_df.empty:
            continue

        grouped = (
            plate_df.groupby(["platecode", "row", "col", "site", "Well"], as_index=False)
            .agg(channel_count=("channel", "nunique"))
            .sort_values(["platecode", "row", "col", "site"])
            .reset_index(drop=True)
        )

        meta_df = load_plate_profile_metadata(
            profiles_root=args.profiles_root.resolve(),
            batch_name=batch_name,
            plate_code=plate_code,
        )
        if "Metadata_Well" in meta_df.columns:
            grouped = grouped.merge(meta_df, left_on="Well", right_on="Metadata_Well", how="left")
        per_plate_grouped.append(grouped)

    if not per_plate_grouped:
        print("No valid filenames found with expected pattern.")
        return

    grouped_all = pd.concat(per_plate_grouped, ignore_index=True)

    complete_df = grouped_all[grouped_all["channel_count"] >= 5].reset_index(drop=True)
    incomplete_df = grouped_all[grouped_all["channel_count"] < 5].reset_index(drop=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = args.output_dir / "labels.csv"
    incomplete_path = args.output_dir / "incomplete_channel_wells.csv"

    complete_df.to_csv(labels_path, index=False)
    incomplete_df.to_csv(incomplete_path, index=False)

    print(f"Saved {len(complete_df)} rows -> {labels_path}")
    print(f"Saved {len(incomplete_df)} rows -> {incomplete_path}")


if __name__ == "__main__":
    main()
