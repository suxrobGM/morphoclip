# CellCLIP Replication Changelog

This document tracks the local changes made to bring `src/cellclip` closer to
the upstream [CellCLIP](https://github.com/suinleelab/CellCLIP) training and
benchmark behavior. (The upstream repo is no longer vendored under `baselines/`.)

It is focused on **replication fidelity**, not general feature development.

## 1. Upstreamish Training Recipe

Added a closer-to-upstream optimization recipe in
[`configs/cellclip/official_upstreamish.yaml`](../../configs/cellclip/official_upstreamish.yaml):

- `lr: 1e-4`
- `weight_decay: 0.1`
- `epochs: 50`
- `warmup_steps: 1000`

This replaced the earlier local-default recipe that was much shorter and more
aggressive:

- `lr: 5e-4`
- `weight_decay: 0.2`
- `epochs: 10`
- `warmup_steps: 100`

## 2. CellCLIP-Style CP-JUMP Split

Added
[`configs/cellclip/official_upstreamish_unique_cpjump_style.yaml`](../../configs/cellclip/official_upstreamish_unique_cpjump_style.yaml)
to run the local trainer on the `cellclip_cpjump_style` split with `eval_subset:
test`.

This matters because the upstream CellCLIP paper path is not equivalent to the
repo's `cpjump1_official_representation` split. The local replication now uses a
deterministic `75/25` split inside each `(Cell_type, Time, Perturbation)` slice,
which is much closer to the upstream baseline script.

## 3. Unique-Perturbation Semantics Fix

The first local implementation of `unique_perturbations` was wrong.

It collapsed examples by `(cell_type, broad_sample)` in
[`src/cellclip/training/dataset.py`](../../src/cellclip/training/dataset.py),
which removed timepoint-distinct examples that upstream CellCLIP would keep.

That produced an over-collapsed run:

- train wells: `1222`
- eval wells: `410`

The corrected behavior is now upstream-faithful for the local feature cache:

- upstream `--unique` strips the last view token from `SAMPLE_KEY`
- local `MorphoCLIPDataset` is already one entry per well bag
- therefore `unique_perturbations=true` should **not** additionally collapse
  across timepoints or perturbation slices

After the fix, the same config resolves to:

- train wells: `10832`
- eval wells: `3408`

## 4. Benchmark Export Fidelity Fix

The earlier benchmark exporter did not use the image tower the same way the
trainer did.

Old exporter behavior in
[`src/cellclip/benchmark/export.py`](../../src/cellclip/benchmark/export.py):

- encode each site with `encode_image`
- average the encoded site embeddings afterward

Corrected exporter behavior:

- first pool site bags with `encode_mil`
- then apply `encode_image`

This now matches the trainer path and the upstream conversion logic.

To support that, the benchmark-only loader was extended in:

- [`src/cellclip/benchmark/model.py`](../../src/cellclip/benchmark/model.py)
- [`src/cellclip/benchmark/checkpoint.py`](../../src/cellclip/benchmark/checkpoint.py)

The benchmark runtime now restores and uses `image_pool` weights instead of only
loading the transformer path.

## 5. Reusable Diagnostics

Added reusable analysis helpers:

- [`src/cellclip/training/analysis.py`](../../src/cellclip/training/analysis.py)
- [`scripts/cellclip/analyze_training_run.py`](../../scripts/cellclip/analyze_training_run.py)

These were added to make replication failures easier to inspect without editing
training code. They report:

- grouped retrieval metrics
- duplicate structure in train/eval splits
- PCA collapse / anisotropy diagnostics
- optional benchmark table comparisons

The duplicate report now includes:

- `unique_plate_wells`
- `unique_cell_broad_samples`
- `unique_cell_time_pert_broad_samples`

This is useful for spotting over-collapsed split logic.

## 6. What Was Misleading Earlier

Two earlier conclusions should not be treated as the final replication result:

1. The old "unique" run looked better in training retrieval because it trained on
   the wrong, over-collapsed dataset.
2. The earlier benchmark numbers for that run were partly distorted by the old
   exporter path.

So the meaningful comparison is:

- corrected training semantics
- corrected benchmark export semantics

not the original local benchmark outputs from before those fixes.

## 7. Current Comparison Snapshot

### Corrected full run

Run directory:

- `output/train_runs/cellclip_upstreamish_cpjump_style_fixed_unique_20260330_145206`

Final training metrics:

- `eval_loss = 7.7498`
- `image_to_text_R@1 = 0.0029`
- `text_to_image_R@1 = 0.0021`

Short benchmark output:

- [`output/benchmark_upstreamish_cpjump_style_fixed_unique_short`](../../output/benchmark_upstreamish_cpjump_style_fixed_unique_short)

Key benchmark numbers:

- Replicability, `compound/A549`: `0.3301`
- Replicability, `compound/U2OS`: `0.3268`
- Replicability, `crispr/A549`: `0.0230`
- Replicability, `crispr/U2OS`: `0.0689`
- Replicability, `orf/U2OS`: `0.0625`
- Matching, `compound/A549`: `0.0577`
- Matching, `compound/U2OS`: `0.0825`
- Matching, `crispr/U2OS`: `0.6667`
- Gene-compound, `compound/crispr U2OS`: `0.0`
- Gene-compound, `compound/orf U2OS`: `0.0`

### Reference pretrained CellCLIP benchmark

Reference benchmark output:

- [`output/benchmark_full_short_cellclip_hf`](../../output/benchmark_full_short_cellclip_hf)

Reference numbers used for comparison:

- Replicability, `compound/A549`: `0.3660`
- Replicability, `compound/U2OS`: `0.3529`
- Replicability, `crispr/A549`: `0.0197`
- Replicability, `crispr/U2OS`: `0.1180`
- Replicability, `orf/U2OS`: `0.1188`
- Matching, `compound/A549`: `0.1780`
- Matching, `compound/U2OS`: `0.1176`
- Matching, `crispr/U2OS`: `1.0000`
- Gene-compound, `compound/crispr U2OS`: `0.0833`

### Practical takeaway

The local replication is now much closer to the reference pretrained CellCLIP on
**compound replicability**, especially:

- `compound/A549`: `0.3301` vs `0.3660`
- `compound/U2OS`: `0.3268` vs `0.3529`

The main remaining gap is now concentrated in:

- within-modality target matching
- cross-modality gene-compound matching
- U2OS CRISPR / ORF quality

## 8. Related Files Added Along the Way

Some files in the same commit support broader CellCLIP work but are not the core
replication fix:

- [`configs/cellclip/official_chemberta_film.yaml`](../../configs/cellclip/official_chemberta_film.yaml)
- [`configs/cellclip/official_chemberta_upstreamish_film_remove_smiles.yaml`](../../configs/cellclip/official_chemberta_upstreamish_film_remove_smiles.yaml)
- [`configs/cellclip/official_chemberta_upstreamish_film_keep_smiles.yaml`](../../configs/cellclip/official_chemberta_upstreamish_film_keep_smiles.yaml)
- [`configs/cellclip/official_chemberta_upstreamish_residual_add.yaml`](../../configs/cellclip/official_chemberta_upstreamish_residual_add.yaml)
- [`configs/cellclip/official_chemberta_upstreamish_concat_mlp.yaml`](../../configs/cellclip/official_chemberta_upstreamish_concat_mlp.yaml)
- [`src/cellclip/training/model.py`](../../src/cellclip/training/model.py)

Those support the ChemBERTa investigation. The ChemBERTa stack now supports:

- matched upstreamish reruns on the corrected recipe
- `film`, `residual_add`, and `concat_mlp` fusion
- `remove_smiles` and `keep_smiles` prompt policies
- frozen, partially tuned, or fully tuned ChemBERTa
- reusable ChemBERTa-aware run diagnostics

They are useful for experiments, but the main upstream-fidelity improvements came from:

- training recipe alignment
- split semantics correction
- benchmark export correction
- reusable diagnostics

## 9. Regression Coverage

The replication fixes are covered by tests in:

- [`tests/cellclip/test_training.py`](../../tests/cellclip/test_training.py)
- [`tests/cellclip/test_analysis.py`](../../tests/cellclip/test_analysis.py)
- [`tests/cellclip/test_benchmark_export.py`](../../tests/cellclip/test_benchmark_export.py)

In particular, the exporter test now protects the trainer-faithful
`encode_mil -> encode_image` path from regressing.
