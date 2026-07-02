"""`morphoclip eval` command: evaluate a trained checkpoint on val/test."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.console import Console
from rich.table import Table

from morphoclip.cli.logging import setup_logging
from morphoclip.data.splits import create_splits
from morphoclip.training.config import load_training_config
from morphoclip.training.engine import autocast_context
from morphoclip.training.evaluate import evaluate_epoch, lookup_text_embeddings
from morphoclip.training.inference import (
    build_eval_dataloader,
    build_eval_dataset,
    filter_batch_to_cached,
    load_models_from_checkpoint,
)
from morphoclip.training.metrics import (
    compute_alignment,
    compute_intra_batch_similarity,
    compute_uniformity,
)
from morphoclip.utils.caching import load_cached_text_features
from morphoclip.utils.device import resolve_device

console = Console()


class Split(StrEnum):
    val = "val"
    test = "test"


def _build_eval_loader(config, device, *, split):
    """Build a DataLoader for the requested split."""
    ds_cfg = config.dataset
    dataset = build_eval_dataset(config)
    _, val_set, test_set = create_splits(
        dataset,
        strategy=ds_cfg.split_strategy,
        val_fraction=ds_cfg.val_fraction,
        seed=config.runtime.seed,
    )
    target = val_set if split == "val" else test_set

    if ds_cfg.max_eval_wells and len(target) > ds_cfg.max_eval_wells:
        target = torch.utils.data.Subset(target, list(range(ds_cfg.max_eval_wells)))

    return build_eval_dataloader(target, config, device), len(target)


def _compute_embedding_diagnostics(
    image_encoder, text_projection, text_cache, loader, *, device, amp
):
    """Compute alignment, uniformity, and intra-batch similarity."""
    image_embs, text_embs = [], []
    skipped = 0
    with torch.no_grad():
        for batch in loader:
            batch, n_skipped = filter_batch_to_cached(batch, text_cache)
            skipped += n_skipped
            if not batch["pert_info"]:
                continue
            features = batch["features"].to(device, non_blocking=True)
            site_mask = batch["site_mask"].to(device, non_blocking=True)
            with autocast_context(device, amp):
                img = image_encoder(features, site_mask)
                raw_text = lookup_text_embeddings(batch["pert_info"], text_cache, device)
                txt = text_projection(raw_text)
            image_embs.append(img.cpu())
            text_embs.append(txt.cpu())
    if skipped:
        console.print(
            f"[yellow]Diagnostics skipped {skipped} wells missing from text cache[/yellow]"
        )

    all_image, all_text = torch.cat(image_embs), torch.cat(text_embs)
    return {
        "alignment": compute_alignment(all_image, all_text),
        "image_uniformity": compute_uniformity(all_image),
        "text_uniformity": compute_uniformity(all_text),
        "image_intra_batch_sim": compute_intra_batch_similarity(all_image),
        "text_intra_batch_sim": compute_intra_batch_similarity(all_text),
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

    image_encoder, text_projection, ckpt, cfg = load_models_from_checkpoint(checkpoint, device)
    if config:
        cfg = load_training_config(str(config))

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
            use_cwa=cfg.optimization.use_cwa,
            amp=cfg.runtime.amp,
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
