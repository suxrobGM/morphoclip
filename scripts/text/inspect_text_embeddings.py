"""Inspect cached text embeddings: cosine similarities before/after projection.

Usage:
    uv run poe inspect-text
    python scripts/text/inspect_text_embeddings.py --device mps
    python scripts/text/inspect_text_embeddings.py --checkpoint path/to/best.pt
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import torch
import torch.nn.functional as F
import typer
from rich.console import Console

from morphoclip.training.inference import load_models_from_checkpoint
from morphoclip.utils.device import resolve_device

CACHE_PATH = "data/text/cached_text_features.pt"
METADATA_DIR = Path("data/metadata/external_metadata")
N = 5  # how many perturbations to inspect

console = Console()


def build_id_labels(metadata_dir: Path) -> dict[str, str]:
    """Map perturbation_id (broad_sample / negcon) -> human-readable label."""
    labels: dict[str, str] = {}

    compound_path = metadata_dir / "JUMP-Target-1_compound_metadata_targets.tsv"
    if not compound_path.exists():
        compound_path = metadata_dir / "JUMP-Target-1_compound_metadata.tsv"
    if compound_path.exists():
        df = pd.read_csv(compound_path, sep="\t")
        name_col = "pert_iname" if "pert_iname" in df.columns else "broad_sample"
        for _, row in df.iterrows():
            pid = str(row.get("broad_sample", "")).strip()
            if not pid:
                continue
            name = str(row.get(name_col, pid)).strip()
            labels[pid] = f"{name} (compound)"

    crispr_path = metadata_dir / "JUMP-Target-1_crispr_metadata.tsv"
    if crispr_path.exists():
        df = pd.read_csv(crispr_path, sep="\t")
        for _, row in df.iterrows():
            pid = str(row.get("broad_sample", "")).strip()
            if not pid:
                continue
            gene = str(row.get("gene", "")).strip() or pid
            labels[pid] = f"{gene} (crispr)"

    orf_path = metadata_dir / "JUMP-Target-1_orf_metadata.tsv"
    if orf_path.exists():
        df = pd.read_csv(orf_path, sep="\t")
        for _, row in df.iterrows():
            pid = str(row.get("broad_sample", "")).strip()
            if not pid:
                continue
            gene = str(row.get("gene", "")).strip() or pid
            labels[pid] = f"{gene} (orf)"

    labels.setdefault("negcon", "negcon (control)")
    return labels


def _print_pairs(title: str, labels: list[str], sim: torch.Tensor) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            console.print(f"  {labels[i]:35s} vs {labels[j]:35s} : {sim[i, j]:.4f}")


def main(
    device: Annotated[str, typer.Option(help="Torch device (default: auto).")] = "auto",
    checkpoint: Annotated[
        Path | None,
        typer.Option(help="Run checkpoint whose trained text projection to apply."),
    ] = None,
) -> None:
    """Inspect cached text embeddings."""
    resolved_device = resolve_device(device)

    cache = torch.load(CACHE_PATH, map_location="cpu")
    emb_raw: torch.Tensor = cache["embeddings"]  # [N_all, 768]
    ids: list[str] = cache["perturbation_ids"]

    emb_raw_small = emb_raw[:N].to(resolved_device)
    ids_small = ids[:N]

    id_labels = build_id_labels(METADATA_DIR)
    pretty = [id_labels.get(pid, pid) for pid in ids_small]

    emb_raw_norm = F.normalize(emb_raw_small, dim=-1)
    sim_raw = emb_raw_norm @ emb_raw_norm.T

    console.rule("[bold blue]Cached text feature inspection")
    console.print(f"Cache: {CACHE_PATH}")
    console.print(f"Device: {resolved_device}")
    console.print(f"Total cached perturbations: {emb_raw.shape[0]}")
    console.print(f"Inspecting first {len(ids_small)} entries")

    console.print("\n[bold]IDs and titles[/bold]")
    for i, (pid, label) in enumerate(zip(ids_small, pretty, strict=False)):
        console.print(f"[{i}] {pid}  ->  {label}", markup=False)

    # An untrained projection head gives random similarities, so this section needs a run.
    if checkpoint is not None:
        loaded = load_models_from_checkpoint(checkpoint, resolved_device)
        with torch.no_grad():
            proj = loaded.text_projection(emb_raw_small)  # [N, 512], L2-normalized
        _print_pairs("Cosine similarities (projected 512-d)", pretty, proj @ proj.T)

    _print_pairs("Cosine similarities (raw 768-d, no projection head)", pretty, sim_raw)


if __name__ == "__main__":
    typer.run(main)
