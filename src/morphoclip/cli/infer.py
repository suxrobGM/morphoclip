"""`morphoclip infer` command: top-k matches, embedding export, or profile export."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.table import Table

from morphoclip.cli.logging import setup_logging
from morphoclip.data.perturbation import PerturbationInfo
from morphoclip.training.config import load_training_config
from morphoclip.training.engine import autocast_context
from morphoclip.training.evaluate import lookup_text_embeddings
from morphoclip.training.inference import (
    build_eval_dataloader,
    build_eval_dataset,
    discover_plates,
    filter_batch_to_cached,
    load_models_from_checkpoint,
)
from morphoclip.utils.caching import load_cached_text_features
from morphoclip.utils.device import resolve_device

console = Console()


class InferMode(StrEnum):
    match = "match"
    embed = "embed"
    profile = "profile"


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


def infer(
    checkpoint: Annotated[Path, typer.Option(help="Path to the trained checkpoint.")],
    config: Annotated[
        Path | None, typer.Option(help="Override the config saved in the checkpoint.")
    ] = None,
    mode: Annotated[InferMode, typer.Option(help="Inference mode.")] = InferMode.match,
    plates: Annotated[
        list[str] | None,
        typer.Option(help="Restrict to specific plates (repeatable). Default: all discovered."),
    ] = None,
    top_k: Annotated[int, typer.Option(help="Top-k matches per well (match mode).")] = 5,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Output directory (default: <run>/infer_<mode>)."),
    ] = None,
    batch_size: Annotated[int | None, typer.Option(help="Override eval batch size.")] = None,
    include_controls: Annotated[
        bool, typer.Option(help="Include control wells in inference.")
    ] = False,
) -> None:
    """Run inference: top-k matches, embedding export, or profile export."""
    setup_logging()

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    device = resolve_device("auto")
    console.rule("[bold blue]MorphoCLIP Inference")
    console.print(f"Checkpoint: {checkpoint} | Device: {device} | Mode: {mode.value}")

    image_encoder, text_projection, ckpt, cfg = load_models_from_checkpoint(checkpoint, device)
    if config:
        cfg = load_training_config(str(config))
    if batch_size:
        cfg.dataset.eval_batch_size = batch_size

    console.print(f"Epoch: {ckpt['epoch']}, step: {ckpt['steps']}")

    text_cache = load_cached_text_features(cfg.dataset.text_cache_path)
    console.print(f"Text cache: {text_cache['embeddings'].shape[0]:,} perturbations")

    resolved_plates = plates or discover_plates(Path(cfg.dataset.feature_root))
    dataset = build_eval_dataset(
        cfg,
        plates=resolved_plates,
        exclude_controls=not include_controls,
    )
    loader = build_eval_dataloader(dataset, cfg, device)
    console.print(f"Plates: {len(resolved_plates)} | Wells: {len(dataset):,}\n")

    console.print("[bold]Encoding wells...[/bold]")
    image_embs, text_embs, all_plates, all_wells, all_pert_infos = _encode_all_wells(
        image_encoder,
        text_projection,
        text_cache,
        loader,
        device=device,
        amp=cfg.runtime.amp,
    )
    console.print(f"Encoded {image_embs.shape[0]} wells -> {image_embs.shape[1]}-d\n")

    resolved_output_dir = output_dir or checkpoint.parent.parent / f"infer_{mode.value}"

    if mode is InferMode.match:
        results = _run_match(
            image_embs,
            text_projection,
            text_cache,
            all_plates,
            all_wells,
            all_pert_infos,
            device=device,
            amp=cfg.runtime.amp,
            top_k=top_k,
        )
        _print_match_summary(results, top_k)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = resolved_output_dir / "matches.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"\nResults saved to [green]{output_path}[/green]")

    elif mode is InferMode.embed:
        _run_embed(
            image_embs,
            text_embs,
            all_plates,
            all_wells,
            all_pert_infos,
            output_dir=resolved_output_dir,
        )

    elif mode is InferMode.profile:
        _run_profile(
            image_embs, all_plates, all_wells, all_pert_infos, output_dir=resolved_output_dir
        )
