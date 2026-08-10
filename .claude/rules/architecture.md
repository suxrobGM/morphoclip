# Architecture

Three top-level packages under `src/`: `morphoclip` (the model), `benchmark`
(evaluation, standalone), `cellclip` (a frozen baseline).

## Layers

Each layer may import from the ones below it, never above.

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

Two things that look like violations and are not:

- `morphoclip.utils.caching` imports `models.prompts.build_prompts` at module
  scope and `models.text_encoder` under `TYPE_CHECKING`. Caching text embeddings
  needs the prompt builder; the encoder itself is only a type.
- `cellclip.training` imports `morphoclip.training.{distributed,metrics,optim,tb_logger}`.
  That is deliberate sharing, not a leak: both trainers use one optimizer and
  schedule builder, and `optim.split_params` takes `decay_first` so CellCLIP
  keeps its param-group order and old checkpoints still resume.

`benchmark` being standalone is enforced by `tests/test_import_graph.py`, which
imports every module in the package and asserts neither torch nor morphoclip
appears in `sys.modules`.

## morphoclip.config

One `StrictModel` base (`extra="forbid"`, `validate_assignment=True`) and one
loader used by both training packages: `load_config(model, path, overrides)`
resolves `extends`, applies `--set` dotted overrides, then validates.

`extends` is resolved outside the model, not in a `model_validator`. A
before-validator has no access to the source path, and re-validating a config
read back from a checkpoint must not re-resolve `extends`.

`tests/config/goldens/` records what every committed config resolves to.
A diff there means resolution changed, which is either the point of the change
or a bug.

## morphoclip.data

- `MetadataIndex` is the only entry point for plate metadata. It returns frozen
  `PerturbationInfo` objects and has no mutators, which is why the test fixture
  is session-scoped.
- `PerturbationType`: `COMPOUND`, `CRISPR`, `ORF`, `NEGCON`, `POSCON`, `UNKNOWN`.
- `MorphoCLIPDataset` samples a *well*: every site in the well stacked into one
  tensor, paired with the well's text. `preload(indices=...)` is partial, so
  `_load_tensor` checks per path rather than treating a non-empty cache as full.
- `config.py` owns `configs/dataset.yml`. Nothing else parses it.
- `pipeline.py` is the unattended extraction loop; `progress.py` is its
  crash-safe resume record.

## morphoclip.models

- `MorphoCLIPTextEncoder`: frozen ModernBERT plus a trainable `ProjectionHead`.
  Do not unfreeze BERT.
- `ProjectionHead`: Linear, LayerNorm, GELU, Dropout, Linear, L2-normalize.
  The output is always L2-normalized.
- `MorphoCLIPImageEncoder` aggregates `(sites, channels, 1024)` into one
  512-d vector. Four aggregators: `ccf-mean` (default), `meanpool-mean`,
  `ccf-attn`, `wellformer`. Sites are an unordered bag, so no aggregator may
  encode site order; channels are ordered, and every aggregator except
  `meanpool` adds a learned per-channel embedding.
- `prompts.py` builds prompt strings from a dict or a `PerturbationInfo`.
  Missing fields become "unknown". There is no `PromptBuilder` class.

## morphoclip.splits

Dataset splitting lives here rather than in `benchmark`, which is what keeps
`benchmark` free of torch.

`strategies.py` holds the strategy table; `api.py` is the entry point
(`create_splits`, `build_split_groups`); `contexts.py` reads the CPJUMP1
reference metadata; `manifest.py` writes the split manifest the benchmark reads.

`build_split_groups` raises for a strategy with no grouping notion rather than
returning `{}`: a grouped consumer that silently receives zero groups falls back
to ungrouped behaviour instead of failing at config-load time.

The `pert_type` bucketing is a deliberate stable md5 on the sample id. Changing
it repartitions every split already on disk.

## benchmark

- `profiles.py` is the on-disk contract: one filename template, and
  `metadata_columns`/`feature_columns` as exact complements, so every column
  lands in exactly one set.
- `data.py` loads and filters profiles; `metrics.py` computes mAP;
  `stable_*.py` is the CPJUMP1 run against the pinned copairs commit
  `880f22a`. "Stable" means the old copairs API, not stability of results.
- `run_with_unpaired_guard` forwards its callee's signature with a ParamSpec.
  It swallows only copairs' `UnpairedException` and the literal "dict_pairs
  empty"; everything else re-raises. Widening it to `except Exception` would
  turn a broken call site into six all-zero result CSVs.
- The harness is not reproducible run to run. Only replicability
  `mean_average_precision` is deterministic; fraction-retrieved varies because
  the matching population is filtered by a random permutation null.

## cellclip

A frozen baseline. Results are in `report/`; keep it runnable, do not extend it.
`cellclip/model.py`'s `QuickGELU`, `ResidualAttentionBlock` and `LayerNorm` are
the OpenAI-CLIP contract the published checkpoint depends on, and
`checkpoint.py` hard-fails on a state-dict mismatch. Do not touch them.
Do not merge CellCLIP components into `morphoclip.models`.

## Baselines

External repos are not vendored. Reference them by link:
[CellCLIP](https://github.com/suinleelab/CellCLIP),
[Chandrasekaran 2024 CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1).
The CPJUMP1 reference metadata the benchmark needs is first-party under
`data/reference/cpjump1/`.
