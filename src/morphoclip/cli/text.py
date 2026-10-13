"""`morphoclip text` command group: precompute."""

import os
from pathlib import Path
from typing import Annotated

import typer

from morphoclip.data.metadata import MetadataIndex
from morphoclip.models.prompts import build_prompt_from_info, prompt_metadata_from_info
from morphoclip.models.text_encoder import MorphoCLIPTextEncoder
from morphoclip.utils.caching import precompute_and_cache_text_embeddings
from morphoclip.utils.console import console
from morphoclip.utils.device import resolve_device

app = typer.Typer(no_args_is_help=True, help="Text embedding pre-computation.")


@app.command()
def precompute(
    config: Annotated[Path, typer.Option(help="Dataset config YAML.")] = Path(
        "configs/dataset.yml"
    ),
    output: Annotated[Path, typer.Option(help="Output cache path.")] = Path(
        "data/text/cached_text_features.pt"
    ),
    batch_size: Annotated[int, typer.Option(help="Encoding batch size.")] = 64,
    device: Annotated[str | None, typer.Option(help="Device (default: auto).")] = None,
    limit: Annotated[int | None, typer.Option(help="Limit perturbations (for testing).")] = None,
    cache_dir: Annotated[Path | None, typer.Option(help="HuggingFace cache directory.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Show prompts only, do not encode.")] = False,
) -> None:
    """Pre-compute and cache raw 768-d BERT text embeddings for all perturbations."""
    if cache_dir is not None:
        resolved_cache_dir = cache_dir.resolve()
        resolved_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(resolved_cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(resolved_cache_dir)

    console.rule("MorphoCLIP Text Embedding Pre-computation")

    console.print(f"Loading metadata from config: {config}")
    index = MetadataIndex.from_config(config)
    console.print(f"  {index}")

    perturbations = index.unique_perturbations()
    if limit:
        perturbations = perturbations[:limit]
    console.print(f"  Unique perturbations: {len(perturbations)}")

    metadata_list = [prompt_metadata_from_info(info) for info in perturbations]

    console.print("\n[bold]Sample prompts:[/bold]")
    for info in perturbations[:5]:
        prompt = build_prompt_from_info(info)
        console.print(f"  [{info.pert_type.value}] {prompt[:100]}...")

    if dry_run:
        console.print("\n[yellow]Dry run -- skipping encoding.[/yellow]")
        return

    resolved_device = str(resolve_device(device or "auto"))
    console.print(f"\nDevice: {resolved_device}")

    encoder = MorphoCLIPTextEncoder(pooling="cls").to(resolved_device)

    total = sum(p.numel() for p in encoder.parameters())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    console.print(f"Total params:     {total:,}")
    console.print(f"Trainable params: {trainable:,}  (projection head only)")

    console.print(f"\nPre-computing and caching to: {output}")
    cache = precompute_and_cache_text_embeddings(
        encoder,
        metadata_list,
        output,
        batch_size=batch_size,
        device=resolved_device,
        show_progress=True,
    )

    console.print(f"\nCached {len(cache['perturbation_ids'])} perturbation embeddings")
    console.print(f"Embedding shape: {cache['embeddings'].shape}")
    console.print(f"Output: {output}")
