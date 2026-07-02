# Supported Split Strategies

This repo now supports three split strategies in `src/morphoclip/data/splits.py`:

- `cpjump1_official_representation`
- `cpjump1_official_gene_compound`
- `cellclip_cpjump_style`

These cover the current benchmark-faithful workflow and the local CellCLIP-style reproduction workflow. Older generic split families were removed to keep the training and evaluation contract small and explicit.

## `cpjump1_official_representation`

Use this for benchmark-faithful representation learning.

- `CRISPR` and `ORF` wells go to `train`
- `Compound` wells at `low` timepoint go to `validate`
- `Compound` wells at `high` timepoint go to `test`

This is the default split for current training configs.

## `cpjump1_official_gene_compound`

Use this for the official target-level gene-compound benchmark split.

- restrict to targets shared across modalities
- assign targets to `train`, `validate`, and `test`
- propagate that target assignment to all linked wells

This is the supported target-holdout option.

## `cellclip_cpjump_style`

Use this to mirror the upstream CellCLIP CP-JUMP train/test recipe more closely.

- form slices by `(Cell_type, Perturbation, Time)`
- within each slice, group by `Metadata_broad_sample`
- assign the first `75%` of sorted groups to `train`
- assign the remaining `25%` to `test`
- `validate` is empty

This is useful for reproducing the CellCLIP-style within-slice holdout rather than the official CPJUMP1 representation split.
