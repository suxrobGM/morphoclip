"""`morphoclip split` command: deterministic train/test split from a labels.csv."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from morphoclip.data.label_splits import create_split_keys, load_and_prepare_labels

console = Console()


def split(
    label_file: Annotated[Path, typer.Option(help="Path to labels.csv.")] = Path(
        "output/labels.csv"
    ),
    output_dir: Annotated[Path, typer.Option(help="Directory for split output CSVs.")] = Path(
        "output/train_test_split"
    ),
    train_ratio: Annotated[
        float, typer.Option(help="Train ratio within each group (deterministic, sorted).")
    ] = 0.75,
    group_columns: Annotated[
        list[str], typer.Option(help="Grouping columns before per-group well split (repeatable).")
    ] = ["batch"],  # noqa: B006 — Typer materializes list defaults per invocation
) -> None:
    """Create a deterministic train/test split from a labels.csv (plate+well key)."""
    full_label_df = load_and_prepare_labels(label_file)
    key_df = full_label_df.drop_duplicates(subset=["UNIQUE_SAMPLE_KEY"]).reset_index(drop=True)

    required_columns = set(group_columns + ["treatment", "UNIQUE_SAMPLE_KEY"])
    missing = required_columns - set(key_df.columns)
    if missing:
        raise ValueError(f"Label table missing required columns: {sorted(missing)}")

    train_keys, test_keys = create_split_keys(key_df, group_columns, train_ratio)
    if set(train_keys) & set(test_keys):
        raise ValueError("Train and test keys overlap.")

    train_label = full_label_df[full_label_df["UNIQUE_SAMPLE_KEY"].isin(train_keys)].copy()
    test_label = full_label_df[full_label_df["UNIQUE_SAMPLE_KEY"].isin(test_keys)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "jumpcp_training_label.csv"
    test_out = output_dir / "jumpcp_testing_label.csv"
    train_label.to_csv(train_out, index=False)
    test_label.to_csv(test_out, index=False)

    console.print(f"Train labels: {len(train_label)}, Test labels: {len(test_label)}")
    console.print(f"Saved: {train_out}")
    console.print(f"Saved: {test_out}")
