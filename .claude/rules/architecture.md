# Architecture

Three top-level packages live under `src/`: `morphoclip` (the model),
`benchmark` (evaluation, standalone), and `cellclip` (a frozen baseline).

## Layers

Each layer may import from the ones below it, never from the ones above.

```
morphoclip.config     pydantic + yaml + pathlib only. No repo imports.
morphoclip.utils      morphoclip.models (one import, see below)
morphoclip.data       morphoclip.{config,utils}
morphoclip.models     morphoclip.data
morphoclip.splits     morphoclip.data
morphoclip.training   morphoclip.{config,data,models,splits,utils}
morphoclip.benchmark  morphoclip.{training,utils}, benchmark.profiles
morphoclip.cli        everything, including cellclip and benchmark
benchmark             pandas/numpy/scipy/copairs. Nothing from morphoclip or cellclip.
cellclip              morphoclip.{config,data,splits,utils}, parts of morphoclip.training,
                      benchmark.{data,profiles,timelines}
```

Two imports look like layer violations but are allowed:

- `morphoclip.utils.caching` imports `models.prompts.build_prompts` at module
  scope, and `models.text_encoder` only under `TYPE_CHECKING`. The text
  embedding cache needs the prompt builder to do its job; the encoder is only
  used as a type hint.
- `cellclip.training` imports `morphoclip.training.{distributed,metrics,optim,tb_logger}`.
  This sharing is deliberate: both trainers use the same optimizer and schedule
  builder. `optim.split_params` takes a `decay_first` flag so CellCLIP keeps
  its original parameter-group order, which lets old checkpoints still resume.

`tests/test_import_graph.py` enforces that `benchmark` stays standalone. It
imports every module in the package and then checks that neither torch nor
morphoclip ended up in `sys.modules`.

## morphoclip.config

One `StrictModel` base class (`extra="forbid"`, `validate_assignment=True`) and
one loader shared by both training packages: `load_config(model, path,
overrides)` resolves `extends`, applies `--set` dotted overrides, then
validates.

`extends` is resolved by the loader, not inside a pydantic `model_validator`.
Two reasons: a before-validator cannot see which file the config came from, and
a config read back out of a checkpoint must not resolve `extends` a second
time.

`tests/config/goldens/` stores the fully resolved form of every committed
config. If a golden changes, config resolution changed. That is either the
point of your change or a bug.

## morphoclip.data

- `MetadataIndex` is the only way in to plate metadata. It returns frozen
  `PerturbationInfo` objects and nothing on it mutates state. That is why the
  test fixture can be session-scoped.
- `PerturbationType`: `COMPOUND`, `CRISPR`, `ORF`, `NEGCON`, `POSCON`, `UNKNOWN`.
- One sample from `MorphoCLIPDataset` is a *well*: all of its sites stacked
  into one tensor, paired with the well's text. `preload(indices=...)` can load
  only part of the dataset, so `_load_tensor` checks each path individually
  instead of assuming a non-empty cache is complete.
- `config.py` owns `configs/dataset.yml`. Nothing else parses that file.
- `pipeline.py` is the unattended feature-extraction loop. `progress.py` is
  its resume record, written so a crash can pick up where it left off.

## morphoclip.models

- `MorphoCLIPTextEncoder`: frozen ModernBERT plus a trainable
  `ProjectionHead`. Do not unfreeze BERT.
- `ProjectionHead`: Linear, LayerNorm, GELU, Dropout, Linear, L2-normalize.
  The output is always L2-normalized.
- `MorphoCLIPImageEncoder` turns `(sites, channels, 1024)` into one 512-d
  vector. Four aggregators: `ccf-mean` (default), `meanpool-mean`, `ccf-attn`,
  `wellformer`. Sites have no meaningful order, so no aggregator may depend on
  site order. Channels do have a fixed order, and every aggregator except
  `meanpool` adds a learned per-channel embedding.
- `prompts.py` builds prompt strings from a dict or a `PerturbationInfo`.
  Missing fields become "unknown". There is no `PromptBuilder` class.

## morphoclip.splits

Splitting lives here, not in `benchmark`. Moving it out is what keeps
`benchmark` free of torch.

`strategies.py` holds the table of split strategies. `api.py` is the entry
point (`create_splits`, `build_split_groups`). `contexts.py` reads the CPJUMP1
reference metadata. `manifest.py` writes the split manifest that the benchmark
reads.

`build_split_groups` raises for a strategy that has no notion of groups,
instead of returning `{}`. If it returned an empty dict, a grouped consumer
would silently run ungrouped instead of failing when the config loads.

The `pert_type` bucketing is a stable md5 hash on the sample id, on purpose.
Changing it moves every sample to a different bucket, which invalidates every
split already saved on disk.

## benchmark

- `profiles.py` defines the on-disk contract: one filename template, and
  `metadata_columns`/`feature_columns` defined as exact complements so every
  column belongs to exactly one of the two sets.
- `data.py` loads and filters profiles. `metrics.py` computes mAP. The
  `stable_*.py` modules run the CPJUMP1 evaluation against the pinned copairs
  commit `880f22a`. "Stable" means the old copairs API, not stable results.
- `run_with_unpaired_guard` keeps its callee's signature via a ParamSpec. It
  catches only copairs' `UnpairedException` and the literal "dict_pairs empty"
  message; everything else re-raises. Do not widen it to `except Exception`:
  that would turn a broken call site into six result CSVs full of zeros.
- The harness gives different numbers on every run. Only the replicability
  `mean_average_precision` is deterministic. Fraction-retrieved varies because
  the set of matched pairs is filtered by a random permutation test.

## cellclip

A frozen baseline. Its results are recorded in `report/`. Keep it runnable,
but do not extend it. In `cellclip/model.py`, the `QuickGELU`,
`ResidualAttentionBlock` and `LayerNorm` classes match the OpenAI-CLIP
implementation that the published checkpoint was trained with, and
`checkpoint.py` fails hard on any state-dict mismatch. Do not touch them.
Do not move CellCLIP components into `morphoclip.models`.

## Baselines

External repos are not copied into this one. Reference them by link:
[CellCLIP](https://github.com/suinleelab/CellCLIP),
[Chandrasekaran 2024 CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1).
The CPJUMP1 reference metadata the benchmark needs is committed to this repo
under `data/reference/cpjump1/`.
