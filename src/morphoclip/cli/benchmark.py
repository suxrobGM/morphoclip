"""`morphoclip benchmark` command (thin wrapper over `benchmark.stable`).

The heavy benchmark stack (copairs, scikit-learn) lives behind the optional
``benchmark`` extra, so it is imported lazily inside the command — the rest of
the CLI stays usable without that extra installed.
"""

from pathlib import Path
from typing import Annotated

import typer

CONFIG_PATH = Path("configs/benchmark.yml")


def benchmark(
    config: Annotated[Path, typer.Option(help="Benchmark config YAML.")] = CONFIG_PATH,
    profiles_dir: Annotated[str | None, typer.Option(help="Override profiles directory.")] = None,
    output_dir: Annotated[str | None, typer.Option(help="Override output directory.")] = None,
    batch: Annotated[str | None, typer.Option(help="Override batch name.")] = None,
    test_mode: Annotated[
        bool | None, typer.Option(help="Use less data for testing (default: from config).")
    ] = None,
    cell_filter: Annotated[str | None, typer.Option(help="Filter to a single cell type.")] = None,
    batch_correction: Annotated[
        bool | None, typer.Option(help="Enable batch correction (default: from config).")
    ] = None,
    pca_kernel: Annotated[str | None, typer.Option(help="PCA kernel for batch correction.")] = None,
    pca_n_components: Annotated[int | None, typer.Option(help="PCA components.")] = None,
    split_manifest: Annotated[str | None, typer.Option(help="Split manifest path.")] = None,
    split_subset: Annotated[
        str | None,
        typer.Option(help="Evaluate one saved split subset (train/val/validate/test)."),
    ] = None,
    timelines: Annotated[
        list[str] | None,
        typer.Option(help="Timeline labels to evaluate: short and/or long (repeatable)."),
    ] = None,
) -> None:
    """Run CPJUMP1 benchmark evaluation (stable copairs mode)."""
    from benchmark.stable import run_stable_benchmark

    run_stable_benchmark(
        config=config,
        profiles_dir=profiles_dir,
        output_dir=output_dir,
        batch=batch,
        test_mode=test_mode,
        cell_filter=cell_filter,
        batch_correction=batch_correction,
        pca_kernel=pca_kernel,
        pca_n_components=pca_n_components,
        split_manifest=split_manifest,
        split_subset=split_subset,
        timelines=timelines,
    )
