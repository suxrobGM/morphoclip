"""`morphoclip eval`, `morphoclip infer`, and `morphoclip split` commands."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import torch
import typer
from rich.console import Console
from rich.table import Table

from morphoclip.cli.logging import setup_logging
from morphoclip.data.perturbation import PerturbationInfo
from morphoclip.data.splits import create_splits
from morphoclip.training.config import load_training_config
from morphoclip.training.engine import autocast_context
from morphoclip.training.evaluate import evaluate_epoch, lookup_text_embeddings
from morphoclip.training.inference import (
    build_eval_dataloader,
    build_eval_dataset,
    discover_plates,
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


class InferMode(StrEnum):
    match = "match"
    embed = "embed"
    profile = "profile"


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# infer
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #


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


def _load_and_prepare_labels(label_file: Path) -> pd.DataFrame:
    full_df = pd.read_csv(label_file)
    full_df["batch"] = _normalize_batch(full_df)
    full_df["Well"] = _normalize_well(full_df)
    full_df["UNIQUE_SAMPLE_KEY"] = full_df["batch"] + "-" + full_df["Well"]
    full_df["SAMPLE_KEY"] = full_df["UNIQUE_SAMPLE_KEY"]
    full_df["treatment"] = full_df["Well"]
    full_df["prompt"] = ""
    return full_df


def _consistent_sample_split(
    group: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = sorted(group["treatment"].dropna().unique().tolist())
    train_size = int(train_ratio * len(ids))
    train_ids = set(ids[:train_size])
    test_ids = set(ids[train_size:])
    return group[group["treatment"].isin(train_ids)], group[group["treatment"].isin(test_ids)]


def _create_split_keys(
    merged_df: pd.DataFrame, group_columns: list[str], train_ratio: float
) -> tuple[list[str], list[str]]:
    train_keys: list[str] = []
    test_keys: list[str] = []
    for _, group in merged_df.groupby(group_columns, dropna=False):
        train_group, test_group = _consistent_sample_split(group, train_ratio)
        train_keys.extend(train_group["UNIQUE_SAMPLE_KEY"].tolist())
        test_keys.extend(test_group["UNIQUE_SAMPLE_KEY"].tolist())
    return train_keys, test_keys


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
    full_label_df = _load_and_prepare_labels(label_file)
    key_df = full_label_df.drop_duplicates(subset=["UNIQUE_SAMPLE_KEY"]).reset_index(drop=True)

    required_columns = set(group_columns + ["treatment", "UNIQUE_SAMPLE_KEY"])
    missing = required_columns - set(key_df.columns)
    if missing:
        raise ValueError(f"Label table missing required columns: {sorted(missing)}")

    train_keys, test_keys = _create_split_keys(key_df, group_columns, train_ratio)
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
