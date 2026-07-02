# MorphoCLIP

Text-supervised contrastive learning for perturbation matching in Cell Painting images.

## Project Overview

MorphoCLIP aligns microscopy image embeddings (DINOv3 CLS tokens) with text descriptions of biological perturbations (compounds, CRISPR knockouts, ORF overexpressions) using contrastive learning. The text encoder is a frozen BioClinical ModernBERT (150M params) with a trainable projection head (768 -> 512).

**Dataset:** CPJUMP1 pilot — 56 plates of Cell Painting images from the Cell Painting Gallery (public S3 bucket).

## Quick Reference

```bash
# One-time setup: install deps + PyTorch for your hardware
uv sync --extra cu128           # (or --extra cu130 / --extra cpu)

# Tasks run through poe (poethepoet): `uv run poe <task>`
uv run poe test                 # Run all tests
uv run poe fetch-dataset        # Download dataset from S3
uv run poe extract-features     # Extract DINOv3 CLS tokens
uv run poe precompute-text      # Pre-compute text embeddings
uv run poe train                # Train the model

# Documentation site (Nextra)
cd docs && bun install          # Install docs dependencies (first time)
cd docs && bun run dev          # Dev server at http://localhost:4000
cd docs && bun run build        # Static export to docs/out/
```

## Package Structure

```text
src/morphoclip/
  data/           # Dataset, metadata, image loading, perturbation types, splits
  models/         # Text encoder, projection head, prompt builder, prompt templates
  utils/          # Text embedding caching, S3 transfer utilities
src/benchmark/    # General benchmark evaluation (metrics, plotting, stable helpers)
src/cellclip/     # CellCLIP baseline (separate from MorphoCLIP)
  benchmark/      # CellCLIP visual encoder, checkpoint loading, export pipeline
  training/       # Local CellCLIP trainer (config, dataset, engine, losses, model)
data/reference/   # First-party CPJUMP1 reference metadata (cpjump1_metadata.csv, JUMP-Target-1 annotations)
scripts/
  data/           # Dataset fetching, plate checks, label generation
  features/       # DINOv3 feature extraction
  text/           # Text embedding pre-computation
  training/       # MorphoCLIP train, eval, inference, train/test split
  benchmark/      # General benchmark scripts (stable benchmark, comparison)
  cellclip/       # CellCLIP-specific scripts (training, export, pipeline)
tests/            # Mirrors src/ structure (data/, models/, benchmark/, cellclip/)
configs/
  dataset.yml     # MorphoCLIP configuration (S3, plates, extraction, splits)
  train/          # MorphoCLIP training configs (base, mean_pool, ddp)
  benchmark.yml   # Benchmark configuration
  cellclip/       # CellCLIP training configs (base + experiment variants)
docs/             # Documentation website (Nextra 4 + Next.js 16.2 + Bun)
  app/            # Next.js App Router (layout, catch-all route)
  content/        # MDX content organized by topic
    getting-started/  # Installation, quick start
    pipeline/         # Training pipeline, feature extraction, text encoder, data fetching
    dataset/          # CPJUMP1 overview, splits, compression
    background/       # Proposal, literature review
    baselines/        # CellCLIP, benchmark guides
  public/         # Static assets (images, diagrams)
  _internal/      # Internal-only docs (not served by Nextra)
```

## Model Architecture

MorphoCLIP aligns image and text embeddings in a shared 512-d L2-normalized space using contrastive learning (CWCL loss + CWA batch correction).

**Image encoder:**

- Frozen DINOv3 ViT-L/16 (300M params, 1024-d CLS tokens)
- 5 fluorescence channels per site -> each replicated to pseudo-RGB -> DINOv3 -> `(5, 1024)` per site
- Features pre-extracted and cached as `.pt` files (~3 GB/plate, ~7 min/plate on RTX 5080)
- CrossChannelFormer (1-layer transformer, 4 heads) aggregates 5 channel tokens into 1 image representation
- Image projection head (1024 -> 512, L2-normalized)
- Alternative: mean pooling (`configs/train/mean_pool.yaml`)

**Text encoder:**

- Frozen BioClinical ModernBERT (150M params, CLS or mean pooling)
- Trainable `ProjectionHead` (768 -> 512, L2-normalized output)
- Verbose prompt templates with compound/gene/CRISPR/ORF/negcon modalities
- Raw 768-d BERT features cached separately from projected 512-d output

**Training:**

- CWCL (Continuously Weighted Contrastive Loss) for soft-positive handling
- CWA (Cross-Well Alignment) for batch effect correction (disabled by default)
- Learnable temperature (LogitScaleModule, clamped to ln(100))
- Trains on both compounds AND genetic perturbations (unlike CellCLIP)
- Default: batch_size=512, lr=1e-4, weight_decay=0.2, 100 epochs
- Target hardware: single RTX 5080 (16 GB VRAM)

## Key Architecture Decisions

- **src-layout**: MorphoCLIP code under `src/morphoclip/`, CellCLIP baseline under `src/cellclip/`, general benchmark under `src/benchmark/`. Scripts are standalone CLI entry points
- **Two template systems**: Verbose templates in `models/prompt_templates.py` (for BERT input), concise labels in `data/perturbation.py` (for dataset display)
- **`PerturbationInfo` bridge**: `PromptBuilder.build_from_info()` connects the data layer to text generation
- **Pre-computed features**: DINOv3 CLS tokens and BERT embeddings are both pre-extracted and cached to enable rapid training iteration (~3-5 min/epoch)

## Development

- **Python**: >=3.14, <3.15
- **Package manager**: uv (single source of truth; PyTorch via `--extra {cpu|cu128|cu130}`)
- **Task runner**: poethepoet (`uv run poe <task>`; tasks defined under `[tool.poe.tasks]`)
- **Test runner**: pytest (`pythonpath = ["src"]` configured in pyproject.toml)
- **Linter**: ruff (`[tool.ruff]` in pyproject.toml)
- **Type checker**: mypy (`mypy_path = "src"`)
- **Config**: `configs/dataset.yml` for MorphoCLIP, `configs/train/` for training, `configs/benchmark.yml` for benchmarks, `configs/cellclip/` for CellCLIP training
- **Docs site**: Nextra 4 + Next.js 16.2 + Bun (in `docs/`)

## Conventions

- Type hints on all public functions
- Docstrings follow Google style (Args/Returns/Raises)
- Imports use full package paths: `from morphoclip.data.metadata import MetadataIndex`, `from cellclip.training.config import ...`
- Scripts use `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))` for imports
- Tests mirror `src/` structure: `tests/data/`, `tests/models/`, `tests/cellclip/`, `tests/benchmark/`
- Commit messages follow conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, etc.
