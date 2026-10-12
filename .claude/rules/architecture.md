# Architecture

## Import layers

A module may import only from layers below it.

```text
morphoclip.config     pydantic, yaml, pathlib only
morphoclip.utils      third-party only (exception below)
morphoclip.data       config, utils
morphoclip.models     data
morphoclip.splits     data
morphoclip.training   config, data, models, splits, utils
morphoclip.benchmark  data, training, utils, benchmark.profiles
morphoclip.cli        anything
benchmark             pandas, numpy, scipy, copairs only
cellclip              morphoclip.{config,data,splits,utils},
                      morphoclip.training.{distributed,metrics,optim,tb_logger},
                      benchmark.{data,profiles,timelines}
```

- Allowed exception: `utils/caching.py` imports `models.prompts` at module
  scope and `models.text_encoder` under `TYPE_CHECKING` only.
- Never import `cellclip` from `morphoclip` library code. Only `morphoclip.cli` may.
- Use absolute imports only. No relative imports in `src/`.

## Import weight

- Keep every `__init__.py` empty. No re-exports.
- Put heavy imports (torch, transformers, pandas, copairs) inside functions
  when the module must stay light. `tests/test_import_graph.py` enforces:
  `morphoclip.training.config` under 400 modules with no torch;
  `benchmark.data` under 900 modules without `benchmark.metrics`.
- If an import-graph test fails, move the import. Do not raise the budget.

## Placement

- Put pipeline logic in `morphoclip.*`, not in `cli/` or `scripts/`.
- Put splitting in `morphoclip.splits`, never in `benchmark`.
- Read plate metadata only through `MetadataIndex`. Parse
  `configs/dataset.yml` only in `morphoclip.data.config`.
- Do not copy external repos (CellCLIP, CPJUMP1) into this one. Link them.
  Committed reference metadata lives in `data/reference/cpjump1/`.

## Invariants

- `MetadataIndex` and `PerturbationInfo` stay immutable. Adding a mutator
  requires narrowing the session-scoped `metadata_index` test fixture.
- `MorphoCLIPDataset.preload(indices=...)` loads subsets, so `_load_tensor`
  must check each path, never assume the cache is complete.
- `build_split_groups` must raise for a strategy without groups. Never return `{}`.
- Look up CWA `PlateOffsets` by barcode via `extract_plate_barcode`, never by
  raw feature-directory name.
- Every aggregator except `meanpool-mean` keeps its learned per-channel embedding.
- `prompts.py` stays plain functions. Missing fields render as "unknown".
