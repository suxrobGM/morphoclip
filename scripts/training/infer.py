"""MorphoCLIP inference script.

Encodes wells through a trained model and computes image-text similarities.
Supports three modes:
  1. **match** (default): Top-k perturbation matches per well
  2. **embed**: Export image/text embeddings as .pt files
  3. **profile**: Export benchmark-compatible well profiles as CSV

Usage:
    uv run poe infer --checkpoint best.pt
    python scripts/training/infer.py --checkpoint best.pt
    python scripts/training/infer.py --checkpoint best.pt --mode embed \
        --output-dir output/embeddings
    python scripts/training/infer.py --checkpoint best.pt --mode profile \
        --output-dir output/profiles
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False, show_path=False)],
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from morphoclip.data.perturbation import PerturbationInfo  # noqa: E402
from morphoclip.training.config import load_training_config  # noqa: E402
from morphoclip.training.engine import autocast_context  # noqa: E402
from morphoclip.training.evaluate import lookup_text_embeddings  # noqa: E402
from morphoclip.training.inference import (  # noqa: E402
    build_eval_dataloader,
    build_eval_dataset,
    discover_plates,
    filter_batch_to_cached,
    load_models_from_checkpoint,
)
from morphoclip.utils.caching import load_cached_text_features  # noqa: E402
from morphoclip.utils.device import resolve_device  # noqa: E402

console = Console()


# --- Encoding ---


def _encode_all_wells(image_encoder, text_projection, text_cache, loader, *, device, amp):
    """Encode all wells and their text through the model."""
    image_embs, text_embs = [], []
    all_plates, all_wells, all_pert_infos = [], [], []
    skipped = 0

    with torch.no_grad():
        for batch in loader:
            batch, n_skipped = filter_batch_to_cached(batch, text_cache)
            skipped += n_skipped
            pert_infos: list[PerturbationInfo] = batch["pert_info"]
            if not pert_infos:
                continue

            features = batch["features"].to(device, non_blocking=True)
            site_mask = batch["site_mask"].to(device, non_blocking=True)

            with autocast_context(device, amp):
                img = image_encoder(features, site_mask)
                raw_text = lookup_text_embeddings(pert_infos, text_cache, device)
                txt = text_projection(raw_text)

            image_embs.append(img.cpu())
            text_embs.append(txt.cpu())
            all_plates.extend(batch["plates"])
            all_wells.extend(batch["wells"])
            all_pert_infos.extend(pert_infos)

    if skipped:
        console.print(f"[yellow]Skipped {skipped} wells with no text cache entry[/yellow]")

    return torch.cat(image_embs), torch.cat(text_embs), all_plates, all_wells, all_pert_infos


# --- Mode runners ---


def _build_unique_text_embeddings(text_projection, text_cache, *, device, amp):
    """Project all cached text embeddings through the trained projection head."""
    raw, ids = text_cache["embeddings"], text_cache["perturbation_ids"]
    projected = []
    with torch.no_grad():
        for i in range(0, raw.shape[0], 256):
            chunk = raw[i : i + 256].to(device, non_blocking=True)
            with autocast_context(device, amp):
                projected.append(text_projection(chunk).cpu())
    return torch.cat(projected), ids


def _find_gt_rank(similarities, pert_ids, ground_truth_id):
    order = torch.argsort(similarities, descending=True)
    for rank, idx in enumerate(order.tolist(), start=1):
        if pert_ids[idx] == ground_truth_id:
            return rank
    return None


def _run_match(
    image_embs, text_projection, text_cache, plates, wells, pert_infos, *, device, amp, top_k
):
    text_embs, pert_ids = _build_unique_text_embeddings(
        text_projection, text_cache, device=device, amp=amp
    )
    sim = image_embs @ text_embs.t()

    results = []
    for i in range(sim.shape[0]):
        scores, indices = torch.topk(sim[i], min(top_k, sim.shape[1]))
        matches = [
            {"perturbation_id": pert_ids[idx], "score": round(s, 6)}
            for s, idx in zip(scores.tolist(), indices.tolist(), strict=True)
        ]
        results.append(
            {
                "plate": plates[i],
                "well": wells[i],
                "ground_truth": pert_infos[i].broad_sample,
                "ground_truth_type": pert_infos[i].pert_type.name,
                "top_matches": matches,
                "rank_of_gt": _find_gt_rank(sim[i], pert_ids, pert_infos[i].broad_sample),
            }
        )
    return results


def _run_embed(image_embs, text_embs, plates, wells, pert_infos, *, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(image_embs, output_dir / "image_embeddings.pt")
    torch.save(text_embs, output_dir / "text_embeddings.pt")

    metadata = [
        {
            "plate": p,
            "well": w,
            "broad_sample": info.broad_sample,
            "pert_type": info.pert_type.name,
        }
        for p, w, info in zip(plates, wells, pert_infos, strict=True)
    ]
    pd.DataFrame(metadata).to_csv(output_dir / "metadata.csv", index=False)
    console.print(
        f"Saved embeddings: image={tuple(image_embs.shape)}, text={tuple(text_embs.shape)}"
    )


def _run_profile(image_embs, plates, wells, pert_infos, *, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    dim = image_embs.shape[1]
    feature_cols = [f"feat_{i}" for i in range(dim)]

    metadata_df = pd.DataFrame(
        {
            "Metadata_Plate": plates,
            "Metadata_Well": wells,
            "Metadata_broad_sample": [info.broad_sample for info in pert_infos],
            "Metadata_pert_type": [info.pert_type.name for info in pert_infos],
            "Metadata_target": [info.target or "" for info in pert_infos],
            "Metadata_gene_symbol": [info.gene_symbol or "" for info in pert_infos],
        }
    )
    features_df = pd.DataFrame(image_embs.numpy(), columns=feature_cols)
    df = pd.concat([metadata_df, features_df], axis=1)

    output_path = output_dir / "morphoclip_profiles.csv.gz"
    df.to_csv(output_path, index=False, compression="gzip")
    console.print(f"Saved profiles: [green]{output_path}[/green] ({len(df)} wells, {dim}-d)")


def _print_match_summary(results, top_k):
    ranks = [r["rank_of_gt"] for r in results if r["rank_of_gt"] is not None]
    if not ranks:
        console.print("[yellow]No ground truth matches found.[/yellow]")
        return

    ranks_t = torch.tensor(ranks, dtype=torch.float)
    table = Table(title="Retrieval Summary", show_lines=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total wells", str(len(results)))
    table.add_row("Wells with GT in cache", str(len(ranks)))
    table.add_row("Mean rank", f"{ranks_t.mean().item():.2f}")
    table.add_row("Median rank", f"{ranks_t.median().item():.1f}")
    for k in (1, 5, 10):
        table.add_row(f"R@{k}", f"{(ranks_t <= k).float().mean().item():.4f}")
    console.print(table)


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(description="MorphoCLIP inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--mode", type=str, default="match", choices=["match", "embed", "profile"])
    parser.add_argument("--plates", type=str, nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--include-controls", action="store_true", default=False)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint_path}[/red]")
        sys.exit(1)

    device = resolve_device("auto")
    console.rule("[bold blue]MorphoCLIP Inference")
    console.print(f"Checkpoint: {checkpoint_path} | Device: {device} | Mode: {args.mode}")

    image_encoder, text_projection, ckpt, config = load_models_from_checkpoint(
        checkpoint_path, device
    )
    if args.config:
        config = load_training_config(args.config)
    if args.batch_size:
        config.dataset.eval_batch_size = args.batch_size

    console.print(f"Epoch: {ckpt['epoch']}, step: {ckpt['steps']}")

    text_cache = load_cached_text_features(config.dataset.text_cache_path)
    console.print(f"Text cache: {text_cache['embeddings'].shape[0]:,} perturbations")

    plates = args.plates or discover_plates(Path(config.dataset.feature_root))
    dataset = build_eval_dataset(
        config,
        plates=plates,
        exclude_controls=not args.include_controls,
    )
    loader = build_eval_dataloader(dataset, config, device)
    console.print(f"Plates: {len(plates)} | Wells: {len(dataset):,}\n")

    console.print("[bold]Encoding wells...[/bold]")
    image_embs, text_embs, all_plates, all_wells, all_pert_infos = _encode_all_wells(
        image_encoder,
        text_projection,
        text_cache,
        loader,
        device=device,
        amp=config.runtime.amp,
    )
    console.print(f"Encoded {image_embs.shape[0]} wells -> {image_embs.shape[1]}-d\n")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else checkpoint_path.parent.parent / f"infer_{args.mode}"
    )

    if args.mode == "match":
        results = _run_match(
            image_embs,
            text_projection,
            text_cache,
            all_plates,
            all_wells,
            all_pert_infos,
            device=device,
            amp=config.runtime.amp,
            top_k=args.top_k,
        )
        _print_match_summary(results, args.top_k)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "matches.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\nResults saved to [green]{output_path}[/green]")

    elif args.mode == "embed":
        _run_embed(
            image_embs, text_embs, all_plates, all_wells, all_pert_infos, output_dir=output_dir
        )

    elif args.mode == "profile":
        _run_profile(image_embs, all_plates, all_wells, all_pert_infos, output_dir=output_dir)


if __name__ == "__main__":
    main()
