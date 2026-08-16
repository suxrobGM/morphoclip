# MorphoCLIP

MorphoCLIP matches Cell Painting microscopy images to text descriptions of the
perturbation applied to the cells (a compound, a CRISPR knockout, or an ORF
overexpression). It trains an image encoder and a text encoder so that matching
pairs land close together in a shared 512-d L2-normalized embedding space.

**Dataset:** CPJUMP1 pilot, 56 plates, downloaded from the public Cell Painting
Gallery S3 bucket.

Layout, import layering and code style live in `.claude/rules/`:
[architecture](.claude/rules/architecture.md),
[project-structure](.claude/rules/project-structure.md),
[coding-conventions](.claude/rules/coding-conventions.md).
This file covers the rest: what the model is, and how to run things.

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

# Paper (needs latexmk and a TeX distribution)
uv run poe report               # build report/build/main.pdf
uv run poe report-arxiv         # pack report/build/arxiv.tar.gz

# Docs site
cd docs && bun install && bun run dev       # http://localhost:4000
```

## Model

**Image branch.** The backbone is a frozen DINOv3 ViT-L/16 (300M params). Each
imaging site has 5 fluorescence channels. Each channel is copied to pseudo-RGB
and encoded on its own, so a site becomes `(5, 1024)` CLS tokens. These tokens
are cached to disk as `.pt` files (about 3 GB and 7 minutes per plate on an
RTX 5080). At train time a small transformer called the CrossChannelFormer
(1 layer, 4 heads) merges the 5 channel tokens into one vector per site, the
site vectors are pooled into one vector per well, and a projection head maps
1024 to 512 with L2 normalization.

Four aggregators are available: `ccf-mean` (default), `meanpool-mean`,
`ccf-attn`, `wellformer`. Sites in a well have no meaningful order, so no
aggregator may depend on site order. Channels do have a fixed order, and every
aggregator except `meanpool` adds a learned per-channel embedding.

**Text branch.** A frozen BioClinical ModernBERT (150M params, CLS pooling by
default) plus a trainable projection head (768 to 512, L2-normalized). Prompts
are built from per-modality templates: compound, gene, CRISPR, ORF, and
negative control. The raw 768-d BERT output is cached separately from the
projected 512-d output, so changing the projection head does not require
re-encoding the text.

**Training.** The loss is CWCL (Continuously Weighted Contrastive Loss): wells
that share a perturbation count as soft positives instead of being pushed
apart. CWA (Cross-Well Alignment) is an optional correction for
plate-to-plate drift, off by default. It subtracts a per-plate offset: the
plate's mean embedding minus the mean over the replicate plates that share its
condition (cell line, experiment type, timepoint). Because the offset is
measured against the condition mean, it removes drift between replicate plates
without removing the condition signal the prompts describe. Offsets are
recomputed at the start of each epoch and saved in the checkpoint. The
temperature is learnable, clamped to ln(100). Unlike CellCLIP, MorphoCLIP
trains on compounds and genetic perturbations together.

`configs/train/base.yaml` sets batch_size=512, lr=1e-4, weight_decay=0.2,
100 epochs, sized for one RTX 5080 (16 GB). The dataclass defaults are smaller
and apply only when no config file is given.

Both encoders read from the pre-computed caches above. With `preload: true`
and the cache in RAM, an epoch at batch size 256 takes about 4 seconds on an
RTX 5080. Loading the cache into memory takes longer than the training run.

## Things worth knowing before you change something

- **The benchmark gives different numbers on every run**, even with identical
  code. Only the replicability `mean_average_precision` is deterministic.
  Fraction-retrieved varies because the set of matched pairs is filtered by a
  random permutation test. Do not treat a small difference as a regression, and
  do not try to make the output byte-identical.
- **CellCLIP is frozen.** Its results are recorded in `report/`. Keep it
  runnable, but do not extend it. Its OpenAI-CLIP layers must stay as they are,
  or the published checkpoint stops loading.
- **Text prompts and dataset labels are two separate systems.** The long
  prompts in `models/prompts.py` feed BERT. The short labels in
  `data/perturbation.py` are for display. `build_prompt_from_info` connects
  the two.
- **Split bucketing uses a stable md5 hash on purpose.** Changing it moves
  every sample to a different split, so every split already saved on disk
  becomes invalid.

## Environment

- Python >=3.14, <3.15; uv is the single source of truth for dependencies
- PyTorch via `--extra {cpu|cu128|cu130}`
- ruff, mypy, pytest and poethepoet are all configured in `pyproject.toml`
- Docs: Nextra 4 + Next.js + Bun, in `docs/`
