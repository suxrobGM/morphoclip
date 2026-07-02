"""Text embedding precomputation and caching utilities.

Provides functions to pre-encode all perturbation text once with the frozen
BioClinical ModernBERT backbone, cache the results, and load them at training
time.  The projection head is NOT included in cached features so that it can
be retrained without invalidating the cache.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from morphoclip.models.prompts import build_prompts

if TYPE_CHECKING:
    from morphoclip.models.text_encoder import MorphoCLIPTextEncoder


def get_perturbation_id(metadata: dict) -> str:
    """Return a stable, unique perturbation ID for joining with image metadata.

    Prefers ``broad_sample`` when present (one per compound/CRISPR guide/ORF).
    Otherwise falls back to compound_name, gene_symbol, or "negcon".
    """
    if metadata.get("broad_sample"):
        return str(metadata["broad_sample"]).strip()
    modality = (metadata.get("modality") or "").lower().strip()
    if modality in ("negcon", "dmso", "control"):
        return "negcon"
    if modality == "compound":
        return (metadata.get("compound_name") or "unknown").strip()
    if modality in ("crispr", "orf"):
        return (metadata.get("gene_symbol") or "unknown").strip()
    return (metadata.get("compound_name") or metadata.get("gene_symbol") or "unknown").strip()


@torch.no_grad()
def _precompute_raw_text_embeddings(
    encoder: MorphoCLIPTextEncoder,
    metadata_list: list[dict],
    batch_size: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
) -> torch.Tensor:
    """Pre-compute raw BERT features (768-d) -- no projection head.

    These are deterministic and stay valid when the projection head is
    retrained.  During training, run ``encoder.projection(cached[k])``
    to get 512-d embeddings.
    """
    encoder = encoder.to(device)
    encoder.eval()

    all_embs: list[torch.Tensor] = []
    n = len(metadata_list)
    iterator = range(0, n, batch_size)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Encoding text (raw BERT)", unit="batch")
        except ImportError:
            pass

    for i in iterator:
        batch_meta = metadata_list[i : i + batch_size]

        prompts = build_prompts(batch_meta, encoder.templates)
        embs = encoder.encode_texts_raw(prompts)  # [B, 768]
        all_embs.append(embs.cpu())

    return torch.cat(all_embs, dim=0)  # [N, 768]


def precompute_and_cache_text_embeddings(
    encoder: MorphoCLIPTextEncoder,
    metadata_list: list[dict],
    output_path: str | Path,
    batch_size: int = 64,
    device: str = "cuda",
    show_progress: bool = True,
) -> dict:
    """Pre-compute raw BERT features and save to a single file.

    Caches BERT output only -- NOT the projection head.  At training time
    run ``encoder.projection(cache["embeddings"][k].unsqueeze(0))`` to get
    512-d embeddings.

    Saves a dict with:
        embeddings:       ``[N, 768]`` tensor (raw BERT [CLS])
        perturbation_ids: list of N unique IDs
        id_to_idx:        dict perturbation_id -> index
        raw:              True
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    for m in metadata_list:
        m["perturbation_id"] = get_perturbation_id(m)

    perturbation_ids = [m["perturbation_id"] for m in metadata_list]
    id_to_idx = {pid: i for i, pid in enumerate(perturbation_ids)}

    raw_embeddings = _precompute_raw_text_embeddings(
        encoder,
        metadata_list,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
    )

    cache = {
        "embeddings": raw_embeddings,
        "perturbation_ids": perturbation_ids,
        "id_to_idx": id_to_idx,
        "raw": True,
    }
    torch.save(cache, output_path)
    return cache


def load_cached_text_features(path: str | Path) -> dict:
    """Load cached text features from disk.

    Returns dict with:
        embeddings:       ``[N, 768]`` tensor
        perturbation_ids: list of N perturbation IDs
        id_to_idx:        dict perturbation_id -> index
        raw:              True if embeddings are 768-d raw (no projection)

    Training-time usage::

        cache = load_cached_text_features("data/cached_text_features.pt")
        k = cache["id_to_idx"]["BRD-K21680..."]
        raw_768 = cache["embeddings"][k]
        text_emb = encoder.projection(raw_768.unsqueeze(0))  # [1, 512]
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "embeddings" not in data:
        raise ValueError(
            "Cached file must be a dict with 'embeddings', 'perturbation_ids', and 'id_to_idx'"
        )
    if "id_to_idx" not in data:
        data["id_to_idx"] = {pid: i for i, pid in enumerate(data["perturbation_ids"])}
    return data
