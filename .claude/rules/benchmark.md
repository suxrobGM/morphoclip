---
paths:
  - "src/benchmark/**"
  - "src/morphoclip/benchmark/**"
  - "src/morphoclip/cli/benchmark.py"
  - "src/morphoclip/cli/export_profiles.py"
  - "configs/benchmark.yml"
  - "tests/benchmark/**"
---

# Benchmark

- `src/benchmark/` is frozen and standalone: pandas, numpy, scipy and copairs
  only. Never import torch, morphoclip or cellclip there.
- `src/morphoclip/benchmark/` (profile export) is active and tested. Keep the
  checkpoint's CWA offsets applied during export.
- Keep `metadata_columns` and `feature_columns` in `profiles.py` exact
  complements.
- Do not upgrade copairs past the pinned commit `880f22a`. The `stable_*.py`
  modules depend on its old API.
- Do not widen `run_with_unpaired_guard`. It may swallow only
  `UnpairedException` and `"dict_pairs empty"`.
- Do not treat run-to-run differences as regressions. Only replicability
  `mean_average_precision` is deterministic.
