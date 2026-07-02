"""Standalone MorphoCLIP evaluation script.

Loads a trained checkpoint, reconstructs models from the saved config,
runs evaluation on val and/or test splits, and saves metrics to JSON.

Usage:
    uv run poe eval --checkpoint best.pt
    python scripts/training/eval.py --checkpoint best.pt --split test
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False, show_path=False)],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from morphoclip.data.splits import create_splits  # noqa: E402
from morphoclip.training.config import load_training_config  # noqa: E402
from morphoclip.training.engine import autocast_context  # noqa: E402
from morphoclip.training.evaluate import evaluate_epoch, lookup_text_embeddings  # noqa: E402
from morphoclip.training.inference import (  # noqa: E402
    build_eval_dataloader,
    build_eval_dataset,
    filter_batch_to_cached,
    load_models_from_checkpoint,
)
from morphoclip.training.metrics import (  # noqa: E402
    compute_alignment,
    compute_intra_batch_similarity,
    compute_uniformity,
)
from morphoclip.utils.caching import load_cached_text_features  # noqa: E402
from morphoclip.utils.device import resolve_device  # noqa: E402

console = Console()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MorphoCLIP model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
        sys.exit(1)

    device = resolve_device("auto")
    console.rule("[bold blue]MorphoCLIP Evaluation")
    console.print(f"Checkpoint: {checkpoint_path} | Device: {device} | Split: {args.split}")

    image_encoder, text_projection, ckpt, config = load_models_from_checkpoint(
        checkpoint_path, device
    )
    if args.config:
        config = load_training_config(args.config)

    logit_scale = torch.nn.Parameter(ckpt["logit_scale"].to(device))
    console.print(
        f"Epoch: {ckpt['epoch']}, step: {ckpt['steps']}, tau: {logit_scale.exp().item():.4f}"
    )

    text_cache = load_cached_text_features(config.dataset.text_cache_path)
    console.print(f"Text cache: {text_cache['embeddings'].shape[0]:,} perturbations")

    loader, n_wells = _build_eval_loader(config, device, split=args.split)
    console.print(f"{args.split.capitalize()} wells: {n_wells:,}\n")

    metrics: dict[str, Any] = dict(
        evaluate_epoch(
            image_encoder,
            text_projection,
            text_cache,
            loader,
            device=device,
            logit_scale=logit_scale,
            loss_type=config.optimization.loss_type,
            use_cwa=config.optimization.use_cwa,
            amp=config.runtime.amp,
        )
    )

    if args.diagnostics:
        console.print("[bold]Computing embedding diagnostics...[/bold]")
        metrics.update(
            _compute_embedding_diagnostics(
                image_encoder,
                text_projection,
                text_cache,
                loader,
                device=device,
                amp=config.runtime.amp,
            )
        )

    metrics.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": ckpt["epoch"],
            "checkpoint_step": ckpt["steps"],
            "split": args.split,
            "n_wells": n_wells,
        }
    )

    table = Table(title=f"Evaluation Results ({args.split} split)", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            table.add_row(key, f"{value:.6f}")
    console.print(table)

    output_path = args.output or str(checkpoint_path.parent.parent / f"eval_{args.split}.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    console.print(f"\nResults saved to [green]{output_path}[/green]")


if __name__ == "__main__":
    main()
