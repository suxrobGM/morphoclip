# Architecture Patterns

## Data Layer (`morphoclip.data`)

- `MetadataIndex` is the single entry point for loading plate metadata and perturbation annotations. Do not duplicate metadata loading elsewhere.
- `PerturbationInfo` dataclass represents a single perturbation (compound/CRISPR/ORF/negcon) with all associated metadata fields.
- `PerturbationType` enum: `COMPOUND`, `CRISPR`, `ORF`, `NEGCON`.
- Image loading goes through `image_loader.py` which handles channel stacking and normalization.

## Models Layer (`morphoclip.models`)

- `MorphoCLIPTextEncoder`: Wraps frozen ModernBERT + trainable `ProjectionHead`. Do not unfreeze BERT weights.
- `ProjectionHead`: Linear -> LayerNorm -> GELU -> Dropout -> Linear -> L2-normalize. Output is always L2-normalized.
- `PromptTemplate`: Dataclass with verbose template strings per modality (compound, crispr, orf, negcon). These are optimized for BERT semantic understanding.
- `PromptBuilder`: Builds text prompts from dicts or `PerturbationInfo` objects. Missing fields default to "unknown".
- The `build_from_info()` bridge maps `PerturbationInfo` fields to template placeholders.

## Utils Layer (`morphoclip.utils`)

- `s3.py`: S3 transfer with AWS CLI or rclone backends. Reusable across scripts.
- `caching.py`: Pre-compute and cache text embeddings. Caches raw 768-d BERT features separately from projected 512-d features.

## CellCLIP Baseline (`cellclip.*`)

CellCLIP is a separate baseline package under `src/cellclip/`, not part of MorphoCLIP. It has two subpackages:

- `cellclip.benchmark`: Visual encoder loading, checkpoint handling, profile export pipeline. Imports from `morphoclip.data` (feature extractor, perturbation) and `benchmark.data`.
- `cellclip.training`: Local CellCLIP trainer with its own config, dataset, engine, losses, and model. Imports from `morphoclip.data` (dataset, metadata, splits) and `benchmark.data`.

CellCLIP uses a different architecture than MorphoCLIP (configurable BERT text encoder, CrossChannelFormer vision, MIL pooling, upstream-style prompts). Do not merge CellCLIP components into `morphoclip.models`.

## General Benchmark (`benchmark.*`)

- `benchmark.data`: Profile loading, split filtering, subset normalization. Shared by both MorphoCLIP and CellCLIP.
- `benchmark.metrics`: mAP computation with copairs backend.
- `benchmark.evaluate`: Benchmark evaluator orchestration.
- `benchmark.plot`: Visualization utilities.
- `benchmark.stable_helpers`: Batch correction helpers.

## Baselines

External comparison repos are not vendored; reference them upstream by link
([CellCLIP](https://github.com/suinleelab/CellCLIP),
[Chandrasekaran 2024 CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1)).
The small CPJUMP1 reference metadata the benchmark harness needs lives first-party
under `data/reference/cpjump1/`.

## Dependency Flow (no cycles)

```
morphoclip.data -> (standalone, no morphoclip deps)
morphoclip.models.prompt_templates -> (standalone)
morphoclip.models.projection_head -> torch
morphoclip.models.prompt_builder -> prompt_templates, data.perturbation
morphoclip.models.text_encoder -> projection_head, prompt_builder
morphoclip.utils.caching -> models.text_encoder (via TYPE_CHECKING)
morphoclip.utils.s3 -> (standalone)
benchmark -> (standalone, uses pandas/numpy/scipy)
cellclip.benchmark -> benchmark.data, morphoclip.data
cellclip.training -> benchmark.data, morphoclip.data, cellclip.benchmark.model
```
