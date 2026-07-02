"""Pre-compute text embeddings for all CPJUMP1 perturbations.

Encodes all perturbation metadata through BioClinical ModernBERT and caches
the raw 768-d features.  During training, run the projection head on cached
features to get 512-d embeddings (cheap MLP forward, no BERT inference).

Usage:
    uv run poe precompute-text
    uv run poe precompute-text --dry-run
    uv run poe precompute-text --limit 10
"""

import argparse
import os
import sys
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from morphoclip.data.metadata import MetadataIndex
from morphoclip.data.perturbation import PerturbationInfo, PerturbationType
from morphoclip.models.prompts import build_prompt_from_info
from morphoclip.models.text_encoder import MorphoCLIPTextEncoder
from morphoclip.utils.caching import precompute_and_cache_text_embeddings
from morphoclip.utils.device import resolve_device

console = Console()


def _unique_perturbations(index: MetadataIndex) -> list[PerturbationInfo]:
    """Extract unique perturbations from the metadata index."""
    seen: set[str] = set()
    result: list[PerturbationInfo] = []
    for _key, info in index._cache.items():
        key = info.broad_sample or f"{info.pert_type.value}:{info.pert_iname or info.gene}"
        if key not in seen:
            seen.add(key)
            result.append(info)
    return result


def _info_to_metadata_dict(info: PerturbationInfo) -> dict:
    """Convert PerturbationInfo to a metadata dict for caching utilities."""
    modality_map = {
        PerturbationType.COMPOUND: "compound",
        PerturbationType.CRISPR: "crispr",
        PerturbationType.ORF: "orf",
        PerturbationType.NEGCON: "negcon",
        PerturbationType.POSCON: "negcon",
        PerturbationType.UNKNOWN: "compound",
    }
    return {
        "modality": modality_map.get(info.pert_type, "compound"),
        "broad_sample": info.broad_sample,
        "compound_name": info.pert_iname,
        "smiles": info.smiles,
        "target_gene": info.target_list,
        "gene_function": info.moa,
        "gene_symbol": info.gene,
        "gene_description": info.protein_name,
        "cell_line": info.cell_line,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings for MorphoCLIP.")
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yml"))
    parser.add_argument("--output", type=Path, default=Path("data/text/cached_text_features.pt"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    parser.add_argument("--limit", type=int, default=None, help="Limit perturbations (for testing)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HuggingFace cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Show prompts only, do not encode")
    args = parser.parse_args()

    if args.cache_dir is not None:
        cache_dir = Path(args.cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)

    console.rule("MorphoCLIP Text Embedding Pre-computation")

    # Load metadata
    console.print(f"Loading metadata from config: {args.config}")
    index = MetadataIndex.from_config(args.config)
    console.print(f"  {index}")

    # Extract unique perturbations
    perturbations = _unique_perturbations(index)
    if args.limit:
        perturbations = perturbations[: args.limit]
    console.print(f"  Unique perturbations: {len(perturbations)}")

    # Convert to metadata dicts
    metadata_list = [_info_to_metadata_dict(info) for info in perturbations]

    # Show sample prompts
    console.print("\n[bold]Sample prompts:[/bold]")
    for info in perturbations[:5]:
        prompt = build_prompt_from_info(info)
        console.print(f"  [{info.pert_type.value}] {prompt[:100]}...")

    if args.dry_run:
        console.print("\n[yellow]Dry run -- skipping encoding.[/yellow]")
        return

    device = str(resolve_device(args.device or "auto"))
    console.print(f"\nDevice: {device}")

    encoder = MorphoCLIPTextEncoder(freeze_bert=True, pooling="cls").to(device)

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    console.print(f"Total params:     {total:,}")
    console.print(f"Trainable params: {trainable:,}  (projection head only)")

    console.print(f"\nPre-computing and caching to: {args.output}")
    cache = precompute_and_cache_text_embeddings(
        encoder,
        metadata_list,
        args.output,
        batch_size=args.batch_size,
        device=device,
        show_progress=True,
    )

    console.print(f"\nCached {len(cache['perturbation_ids'])} perturbation embeddings")
    console.print(f"Embedding shape: {cache['embeddings'].shape}")
    console.print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
