# CellCLIP training configs

Five files, because every other variant differed from one of these by a handful
of keys and is reproducible with `--set`. CellCLIP is a frozen baseline: its
results are in `report/`, and the runs behind them are listed below.

## The files

| File | What it is |
| --- | --- |
| `base.yaml` | Shared defaults. Everything else extends this. |
| `official_baseline.yaml` | The official-split recipe. Default for `morphoclip cellclip train`. |
| `cellclip_style_base.yaml` | Same recipe on a CellCLIP-style CP-JUMP1 split: deterministic 75/25 train/test inside each `(Cell_type, Time, Perturbation)` slice, grouped by `broad_sample`. |
| `official_upstreamish.yaml` | Upstream's larger batch, lower learning rate, longer warmup and longer run. |
| `official_upstreamish_unique_cpjump_style.yaml` | The upstream recipe on the CellCLIP-style split. Parent of every ChemBERTa variant below. |

Split strategies available: `cpjump1_official_representation`,
`cpjump1_official_gene_compound`, `cellclip_cpjump_style`.

## Reproducing the historical runs

Each command below reproduces a config that used to live in this directory. The
overrides were computed from the resolved configs, not transcribed, so they are
exact. `tests/config/goldens/` pins what the surviving five resolve to.

ChemBERTa FiLM, SMILES removed from prompts:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/official_upstreamish_unique_cpjump_style.yaml \
  --set model.variant=chemberta_film \
  --run-name cellclip_chemberta_upstreamish_film_remove_smiles
```

ChemBERTa FiLM, SMILES kept in prompts:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/official_upstreamish_unique_cpjump_style.yaml \
  --set model.variant=chemberta \
  --set model.chem_prompt_policy=keep_smiles \
  --run-name cellclip_chemberta_upstreamish_film_keep_smiles
```

Concat-MLP fusion instead of FiLM:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/official_upstreamish_unique_cpjump_style.yaml \
  --set model.variant=chemberta \
  --set model.chem_prompt_policy=keep_smiles \
  --set model.chem_fusion_type=concat_mlp \
  --run-name cellclip_chemberta_upstreamish_concat_mlp
```

Residual-add fusion instead of FiLM:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/official_upstreamish_unique_cpjump_style.yaml \
  --set model.variant=chemberta \
  --set model.chem_prompt_policy=keep_smiles \
  --set model.chem_fusion_type=residual_add \
  --run-name cellclip_chemberta_upstreamish_residual_add
```

CLS pooling, unfrozen ChemBERTa, site augmentation:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/official_upstreamish_unique_cpjump_style.yaml \
  --set model.variant=chemberta \
  --set model.chem_prompt_policy=keep_smiles \
  --set model.chemberta_pooling=cls \
  --set model.freeze_chemberta=false \
  --set dataset.train_max_sites_per_well=5 \
  --set dataset.within_well_interp_sites=1 \
  --set dataset.same_pert_interp_sites=1 \
  --run-name cellclip_chemberta_upstreamish_film_keep_cls_augmented
```

ChemBERTa FiLM on the official recipe rather than the upstream one:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/official_baseline.yaml \
  --set model.variant=chemberta_film \
  --set dataset.split_strategy=cellclip_cpjump_style \
  --set dataset.eval_subset=test \
  --run-name cellclip_chemberta_film_cpjump_style_full
```

1024-d DINOv3 ViT-L features instead of the default cache:

```bash
uv run morphoclip cellclip train \
  --config configs/cellclip/base.yaml \
  --set dataset.feature_root=data/features \
  --set model.vision_width=1024 \
  --run-name cellclip_1024dim_baseline
```

`cellclip_jumpcp.yaml` resolved to exactly `official_baseline.yaml`, so use that
file directly.
