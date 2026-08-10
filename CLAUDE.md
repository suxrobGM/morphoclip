# MorphoCLIP

Text-supervised contrastive learning for perturbation matching in Cell Painting images.

MorphoCLIP aligns microscopy image embeddings with text descriptions of biological
perturbations (compounds, CRISPR knockouts, ORF overexpressions) in a shared 512-d
L2-normalized space.

**Dataset:** CPJUMP1 pilot, 56 plates from the Cell Painting Gallery (public S3 bucket).

Layout, layering and code style live in `.claude/rules/`:
[architecture](.claude/rules/architecture.md),
[project-structure](.claude/rules/project-structure.md),
[coding-conventions](.claude/rules/coding-conventions.md).
This file covers what those do not: the model itself, and how to run things.

## Quick reference

```bash
# One-time setup: install deps + PyTorch for your hardware
uv sync --extra cu128           # (or --extra cu130 / --extra cpu)

# Every pipeline step is a `morphoclip` subcommand
uv run morphoclip --help                    # list all commands and groups
uv run morphoclip data fetch                # download dataset from S3
uv run morphoclip features extract          # extract DINOv3 CLS tokens
uv run morphoclip text precompute           # pre-compute text embeddings
uv run morphoclip train --config configs/train/base.yaml
uv run morphoclip eval --checkpoint <path>
uv run morphoclip benchmark --config configs/benchmark.yml

# Multi-GPU
uv run torchrun --nproc_per_node=4 -m morphoclip.cli train \
  --config configs/train/ddp.yaml --distributed

# Dev loop (poethepoet)
uv run poe check                # format-check, lint, typecheck, test
uv run poe test

# Docs site
cd docs && bun install && bun run dev       # http://localhost:4000
```

## Model

**Image branch.** Frozen DINOv3 ViT-L/16 (300M params). Each site's 5 fluorescence
channels are replicated to pseudo-RGB and encoded to `(5, 1024)` CLS tokens, cached
to disk as `.pt` (about 3 GB and 7 minutes per plate on an RTX 5080). At train time a
CrossChannelFormer (1 layer, 4 heads) collapses the 5 channel tokens into one image
representation, sites are pooled, and a projection head maps 1024 to 512 with
L2-normalization.

Four aggregators are available: `ccf-mean` (default), `meanpool-mean`, `ccf-attn`,
`wellformer`. Sites within a well are an unordered bag; channels are ordered, and
every aggregator except `meanpool` adds a learned per-channel embedding.

**Text branch.** Frozen BioClinical ModernBERT (150M params, CLS pooling by default)
plus a trainable projection head (768 to 512, L2-normalized). Prompts come from
verbose per-modality templates covering compound, gene, CRISPR, ORF and negcon. The
raw 768-d BERT features are cached separately from the projected 512-d output, so
changing the projection does not mean re-encoding text.

**Training.** CWCL (Continuously Weighted Contrastive Loss) handles soft positives;
CWA (Cross-Well Alignment) corrects batch effects and is off by default. Temperature
is learnable, clamped to ln(100). Unlike CellCLIP, MorphoCLIP trains on compounds and
genetic perturbations together.

`configs/train/base.yaml` sets batch_size=512, lr=1e-4, weight_decay=0.2, 100 epochs,
sized for a single RTX 5080 (16 GB). The dataclass fallbacks are smaller and only
apply when no config is given.

Both encoders read from pre-extracted caches, which is what makes a 3 to 5 minute
epoch possible.

## Things worth knowing before you change something

- **The benchmark is not reproducible run to run.** Two runs of identical code differ
  in all six result CSVs. Only replicability `mean_average_precision` is
  deterministic; fraction-retrieved varies because the matching population is
  filtered by a random permutation null. Do not treat a small delta as a regression,
  and do not chase byte-identical output.
- **CellCLIP is frozen.** Its results are in `report/`. Keep it runnable; do not
  extend it. Its OpenAI-CLIP layers are a checkpoint contract.
- **Text prompts and dataset labels are two systems.** Verbose prompts in
  `models/prompts.py` feed BERT; concise labels in `data/perturbation.py` are for
  display. `build_prompt_from_info` bridges them.
- **Split bucketing is a deliberate stable md5.** Changing it repartitions every
  split already on disk.

## Environment

- Python >=3.14, <3.15; uv is the single source of truth for dependencies
- PyTorch via `--extra {cpu|cu128|cu130}`
- ruff, mypy, pytest and poethepoet all configured in `pyproject.toml`
- Docs: Nextra 4 + Next.js + Bun, in `docs/`
