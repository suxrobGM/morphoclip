"""`morphoclip eval` command: evaluate a trained checkpoint on val/test."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.table import Table

from morphoclip.splits.api import create_splits
from morphoclip.splits.strategies import SplitParams
from morphoclip.training.config import load_training_config
from morphoclip.training.evaluate import evaluate_epoch
from morphoclip.training.inference import (
    build_eval_dataloader,
    build_eval_dataset,
    encode_wells,
    load_models_from_checkpoint,
)
from morphoclip.training.metrics import (
    compute_alignment,
    compute_intra_batch_similarity,
    compute_uniformity,
)
from morphoclip.utils.caching import load_cached_text_features
from morphoclip.utils.console import console, setup_logging
from morphoclip.utils.device import resolve_device


class Split(StrEnum):
    val = "val"
    test = "test"


def _build_eval_loader(config, device, *, split):
    """Build a DataLoader for the requested split."""
    ds_cfg = config.dataset
    dataset = build_eval_dataset(config)
    _, val_set, test_set = create_splits(
        dataset,
        ds_cfg.split_strategy,
        SplitParams(val_fraction=ds_cfg.val_fraction, seed=config.runtime.seed),
    )
    target = val_set if split == "val" else test_set

    return build_eval_dataloader(target, config, device), len(target)


def _compute_embedding_diagnostics(
    image_encoder, text_projection, text_cache, loader, *, device, amp, plate_offsets
) -> dict[str, float]:
    """Compute alignment, uniformity, and intra-batch similarity."""
    encoded = encode_wells(
        image_encoder,
        loader,
        device=device,
        text_projection=text_projection,
        text_cache=text_cache,
        amp=amp,
        plate_offsets=plate_offsets,
    )
    if encoded.skipped:
        console.print(
            f"[yellow]Diagnostics skipped {encoded.skipped} wells missing from text cache[/yellow]"
        )

    image, text = encoded.image, encoded.require_text()
    return {
        "alignment": compute_alignment(image, text),
        "image_uniformity": compute_uniformity(image),
        "text_uniformity": compute_uniformity(text),
        "image_intra_batch_sim": compute_intra_batch_similarity(image),
        "text_intra_batch_sim": compute_intra_batch_similarity(text),
    }


def evaluate(
    checkpoint: Annotated[Path, typer.Option(help="Path to the trained checkpoint.")],
    config: Annotated[
        Path | None,
        typer.Option(help="Override the config saved in the checkpoint."),
    ] = None,
    split: Annotated[Split, typer.Option(help="Which split to evaluate.")] = Split.val,
    output: Annotated[
        Path | None,
        typer.Option(help="Output JSON path (default: <run>/eval_<split>.json)."),
    ] = None,
    diagnostics: Annotated[
        bool,
        typer.Option(help="Also compute alignment/uniformity/intra-batch diagnostics."),
    ] = False,
) -> None:
    """Evaluate a trained MorphoCLIP checkpoint on the val or test split."""
    setup_logging()

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    device = resolve_device("auto")
    console.rule("[bold blue]MorphoCLIP Evaluation")
    console.print(f"Checkpoint: {checkpoint} | Device: {device} | Split: {split.value}")

    loaded = load_models_from_checkpoint(checkpoint, device)
    image_encoder, text_projection = loaded.image_encoder, loaded.text_projection
    plate_offsets, ckpt = loaded.plate_offsets, loaded.ckpt
    cfg = load_training_config(str(config)) if config else loaded.config

    logit_scale = torch.nn.Parameter(ckpt["logit_scale"].to(device))
    console.print(
        f"Epoch: {ckpt['epoch']}, step: {ckpt['steps']}, tau: {logit_scale.exp().item():.4f}"
    )

    text_cache = load_cached_text_features(cfg.dataset.text_cache_path)
    console.print(f"Text cache: {text_cache['embeddings'].shape[0]:,} perturbations")

    loader, n_wells = _build_eval_loader(cfg, device, split=split.value)
    console.print(f"{split.value.capitalize()} wells: {n_wells:,}\n")

    metrics: dict[str, Any] = dict(
        evaluate_epoch(
            image_encoder,
            text_projection,
            text_cache,
            loader,
            device=device,
            logit_scale=logit_scale,
            loss_type=cfg.optimization.loss_type,
            amp=cfg.runtime.amp,
            plate_offsets=plate_offsets,
            target_weight=cfg.optimization.target_weight,
        )
    )

    if diagnostics:
        console.print("[bold]Computing embedding diagnostics...[/bold]")
        metrics.update(
            _compute_embedding_diagnostics(
                image_encoder,
                text_projection,
                text_cache,
                loader,
                device=device,
                amp=cfg.runtime.amp,
                plate_offsets=plate_offsets,
            )
        )

    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": ckpt["epoch"],
            "checkpoint_step": ckpt["steps"],
            "split": split.value,
            "n_wells": n_wells,
        }
    )

    table = Table(title=f"Evaluation Results ({split.value} split)", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
    console.print(table)

    output_path = output or checkpoint.parent.parent / f"eval_{split.value}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    console.print(f"\nResults saved to [green]{output_path}[/green]")
