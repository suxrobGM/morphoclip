# CellCLIP Training Config Variants

These config files are meant for controlled comparisons of the local CellCLIP
training recipe without editing YAML in place.

Supported split strategies in the repo are now:

- `cpjump1_official_representation`
- `cpjump1_official_gene_compound`
- `cellclip_cpjump_style`

- `base.yaml`
  - Shared defaults for the local CellCLIP trainer.
- `official_baseline.yaml`
  - Current official-split recipe used by `configs/cellclip/cellclip_jumpcp.yaml`.
- `cellclip_style_base.yaml`
  - Same local training recipe, but with a CellCLIP-style CP-JUMP1 split:
    deterministic `75/25` train/test inside each `(Cell_type, Time, Perturbation)`
    slice, grouped by `broad_sample`.
- `official_batch64.yaml`
  - Isolates batch-size effects relative to the baseline.
- `official_low_lr_long_warmup.yaml`
  - Isolates a slower optimization schedule closer to the upstream recipe.
- `official_upstreamish.yaml`
  - Combines the larger batch, lower learning rate, longer warmup, and longer run.
- `official_upstreamish_unique_cpjump_style.yaml`
  - Uses the upstreamish optimization recipe with `cellclip_cpjump_style`.
    On the local well-bag feature cache, `unique_perturbations` is effectively a
    no-op because each dataset entry is already one well-level bag.
- `official_chemberta_film.yaml`
  - Keeps the local CellCLIP training recipe, removes SMILES from compound prompts,
    and adds frozen ChemBERTa SMILES conditioning through FiLM on the prompt branch.
- `official_chemberta_upstreamish_film_remove_smiles.yaml`
  - Matched-recipe ChemBERTa rerun on the corrected upstreamish schedule, preserving
    the original ChemBERTa FiLM setup with SMILES removed from compound prompts.
- `official_chemberta_upstreamish_film_keep_smiles.yaml`
  - Same upstreamish ChemBERTa recipe, but keeps SMILES in the prompt while still
    conditioning through ChemBERTa FiLM.
- `official_chemberta_upstreamish_residual_add.yaml`
  - Uses the matched upstreamish recipe with ChemBERTa residual-add fusion.
- `official_chemberta_upstreamish_concat_mlp.yaml`
  - Uses the matched upstreamish recipe with ChemBERTa concat-MLP fusion.
- `schedules/chemberta_full_benchmark.yaml`
  - Declarative 3-stage ChemBERTa sweep spec used by
    `scripts/cellclip/run_scheduler.py` for train-analysis-benchmark scheduling.
- `schedules/chemberta_same_pert_aggressive.yaml`
  - Short-benchmark ChemBERTa sweep that keeps the best FiLM+keep-smiles+cls
    recipe fixed and stresses same-perturbation interpolation at `3`, `7`, and
    `11` replacement slots with a 9-site training bag.

Example:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/cellclip/train_cellclip.py \
  --config configs/cellclip/official_upstreamish.yaml \
  --run-name cellclip_official_upstreamish
```
