# Same-Perturbation Interpolation Experiment

This note summarizes the aggressive same-perturbation interpolation sweep run on
top of the current best ChemBERTa CellCLIP recipe.

## Setup

Base recipe:

- `variant: chemberta`
- `chem_fusion_type: film`
- `chem_prompt_policy: keep_smiles`
- `chemberta_pooling: cls`
- full ChemBERTa tuning (`freeze_chemberta: false`, `chemberta_tune_layers: 0`)
- `cellclip_cpjump_style` split
- upstreamish optimization schedule

Experiment change:

- keep `within_well_interp_sites: 0`
- raise `train_max_sites_per_well` from `5` to `9`
- sweep `same_pert_interp_sites` over `3`, `7`, and `11`
- run `short` benchmark only

Sweep spec:

- [`configs/cellclip/schedules/chemberta_same_pert_aggressive.yaml`](../../configs/cellclip/schedules/chemberta_same_pert_aggressive.yaml)

Stage 3 runs:

- `same_pert_3__full`
- `same_pert_7__full`
- `same_pert_11__full`

Reference comparison runs:

- previous best augmentation run:
  [`output/benchmark_chemberta_full_augmentation_stage3_rerun__same_pert_only__full__stage3__full`](../../output/benchmark_chemberta_full_augmentation_stage3_rerun__same_pert_only__full__stage3__full)
- full baseline:
  [`output/benchmark_full_baseline`](../../output/benchmark_full_baseline)
- pretrained CellCLIP:
  [`output/benchmark_full_cellclip_hf`](../../output/benchmark_full_cellclip_hf)

## Internal Sweep Ranking

Scheduler internal ranking uses grouped retrieval plus PCA collapse tie-breaks.

Stage 1:

1. `same_pert_7`
2. `same_pert_11`
3. `same_pert_3`

Stage 2:

1. `same_pert_7__full`
2. `same_pert_11__full`
3. `same_pert_3__full`

Stage 3:

1. `same_pert_11__full`
2. `same_pert_3__full`
3. `same_pert_7__full`

Stage 3 leaderboard:

- [`output/sweeps/chemberta_same_pert_aggressive/leaderboard_stage3.csv`](../../output/sweeps/chemberta_same_pert_aggressive/leaderboard_stage3.csv)

This ranking should not be confused with the final benchmark outcome. The actual
short benchmark favors a more conservative interpretation.

## Short Benchmark Results

### Replicability

| Metric              | same_pert_3 | same_pert_7 | same_pert_11 | old best same_pert_only |
| ------------------- | ----------: | ----------: | -----------: | ----------------------: |
| compound short A549 |      0.3954 |      0.3562 |       0.3693 |                  0.3922 |
| compound short U2OS |      0.3105 |      0.3203 |       0.2908 |                  0.3268 |
| crispr short A549   |      0.2262 |      0.0852 |       0.0033 |                  0.1016 |
| crispr short U2OS   |      0.0623 |      0.0689 |       0.0197 |                  0.0590 |
| orf short A549      |      0.0063 |      0.0000 |       0.0000 |                  0.0000 |
| orf short U2OS      |      0.0688 |      0.0750 |       0.0875 |                  0.0688 |

### Matching

| Metric              | same_pert_3 | same_pert_7 | same_pert_11 | old best same_pert_only |
| ------------------- | ----------: | ----------: | -----------: | ----------------------: |
| compound short A549 |      0.1575 |      0.0536 |       0.3101 |                  0.2362 |
| compound short U2OS |      0.0217 |      0.0673 |       0.1705 |                  0.1524 |
| crispr short A549   |      0.2727 |      0.3333 |            - |                  0.0000 |
| crispr short U2OS   |      1.0000 |      0.3333 |       1.0000 |                  1.0000 |

### Gene-Compound

| Metric                | same_pert_3 | same_pert_7 | same_pert_11 | old best same_pert_only |
| --------------------- | ----------: | ----------: | -----------: | ----------------------: |
| compound->crispr A549 |      0.0000 |      0.0000 |            - |                  0.0263 |
| compound->crispr U2OS |      0.0000 |      0.0000 |       0.0000 |                  0.0000 |
| compound->orf A549    |      0.0000 |           - |            - |                       - |
| compound->orf U2OS    |      0.1667 |      0.0000 |       0.0000 |                  0.0000 |

Benchmark roots:

- [`same_pert_3`](../../output/benchmark_chemberta_same_pert_aggressive__same_pert_3__full__short)
- [`same_pert_7`](../../output/benchmark_chemberta_same_pert_aggressive__same_pert_7__full__short)
- [`same_pert_11`](../../output/benchmark_chemberta_same_pert_aggressive__same_pert_11__full__short)

## Interpretation

### `same_pert_3`

This is the most balanced aggressive setting.

Strengths:

- slightly improves `compound short A549` replicability over the old best
- strongly improves `crispr short A549` replicability
- produces a non-zero `compound->orf U2OS` gene-compound score

Weaknesses:

- compound matching is worse than the old best on both A549 and U2OS
- does not improve broad compound replicability enough to justify replacing the
  simpler `same_pert_only` setting

### `same_pert_7`

This setting is not useful.

It looked good on internal scheduler metrics, but on the actual short benchmark
it regressed compound replicability and compound matching relative to the old
best ChemBERTa variant.

### `same_pert_11`

This is an overmixed compound-matching specialist.

Strengths:

- best `compound short A549` matching in the sweep
- best `compound short U2OS` matching in the sweep

Weaknesses:

- severe collapse in CRISPR short replicability
- weaker compound replicability than `same_pert_3`
- does not look like a safe general-purpose replacement

One implementation detail matters here: with `train_max_sites_per_well: 9`,
`same_pert_interp_sites: 11` saturates at the number of available site slots.
So this run should be interpreted as a near-maximal same-perturbation mixing
stress test, not as a truly distinct 11-site interpolation regime.

## Low-Support CRISPR Matching Warning

The large CRISPR matching fractions in these runs are still low-support.

Support counts from `cellprofiler_matching_map.csv`:

- `same_pert_3`
  - `A549 short`: `3 / 11`
  - `U2OS short`: `2 / 2`
- `same_pert_7`
  - `A549 short`: `1 / 3`
  - `U2OS short`: `1 / 3`
- `same_pert_11`
  - `U2OS short`: `2 / 2`

So the `1.0` rows should be treated as fragile and denominator-limited, not as
robust wins.

## Conclusion

The aggressive sweep does not produce a clean replacement for the earlier best
ChemBERTa augmentation variant `same_pert_only__full`.

Practical takeaway:

- keep `same_pert_only__full` as the default best overall variant
- keep `same_pert_3` as the only aggressive setting worth future follow-up
- drop `same_pert_7`
- treat `same_pert_11` as evidence that overmixing can improve some compound
  matching rows while damaging broader retrieval and replicability

If another interpolation pass is attempted, it should start from `same_pert_3`
and vary only one factor at a time, ideally with reduced CPU-side data loading
overhead in the augmentation path.
