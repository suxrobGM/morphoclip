# Cleanup Handoff — post uv migration

The PDM → uv + poe migration (see `.claude/plans/migrate-project-to-uv-*.md`) intentionally left the items
below untouched to keep that change focused. This doc is the pickup list for a dedicated cleanup session
(the user flagged an intent to remove redundant modules / files / external deps / submodules).

## 1. Benchmark reproducibility conda path

- `benchmark_environment.yml` (py3.10 conda env "analysis"; installs `copairs` from a pinned git commit)
  still exists. `src/benchmark/{metrics,evaluate,stable_helpers}.py` import `copairs`, which is **not** in
  `pyproject.toml` — so benchmark eval currently only runs inside that separate conda env, never in the uv env.
- Decision needed:
  - (a) Fold `copairs` into uv as an optional `benchmark` extra
    (`copairs @ git+https://github.com/cytomining/copairs`), verify it resolves on Python 3.14, then delete
    `benchmark_environment.yml`; **or**
  - (b) Confirm the benchmark is no longer required and delete `src/benchmark`'s copairs usage
    (and possibly the whole `benchmark` package + `src/cellclip` benchmark stages) plus the env file.
- `docs/content/baselines/benchmark-reproducibility.mdx` still documents `conda env create -f
  benchmark_environment.yml` / `conda activate analysis` / `conda install rclone` and a `.venv/bin/python`
  reference — rewrite or delete alongside the decision above.

## 2. `micromamba` calls in the CellCLIP scheduler

- `src/cellclip/scheduler.py` still shells out to `micromamba run -n cpjump python …` for the export
  (`export_cellclip_profiles.py`) and benchmark (`benchmark_stable.py`) stages (~lines 379–412). These depend
  on the benchmark conda env in (1). Once (1) is resolved, switch them to `uv run` (or delete if the
  benchmark path is cut). The training/analysis subprocess calls in this file were already migrated to
  `uv run` during the uv migration.

## 3. `baselines/` submodules and cellprofiler

- `baselines/CellCLIP` and `baselines/2024_Chandrasekaran_NatureMethods_CPJUMP1` are git submodules with
  their own conda/HPC/`requirements.txt` files (upstream — not edited in place).
- `baselines/cellprofiler/` is a local (non-submodule) conda-based baseline (`environment.yml`, README).
- Decision needed: which of these submodules/baselines are still required? If any are dropped, also purge the
  corresponding `.gitmodules` entries and any imports/docs that reference them.

## 4. Misc redundant-module audit

- After the removals above, sweep `src/` for modules / dead code no longer reachable (e.g. anything only used
  by the benchmark or CellCLIP-scheduler export paths) and prune. Re-run `uv run poe test`,
  `uv run poe lint`, and `uv run poe typecheck` after each removal.

## Reference: what the uv migration already did

- `pyproject.toml`: PDM blocks → `[tool.uv]` (torch extras `cpu`/`cu128`/`cu130`), `[dependency-groups].dev`
  (adds `poethepoet`), `[tool.poe.tasks]`, `[tool.ruff]`.
- Deleted: `pdm.lock`, `pdm.toml`, `.pdm-python`, `environment.yml` (main conda env).
- Added: `.python-version` (3.14), `uv.lock`.
- Migrated functional refs: `src/cellclip/scheduler.py` (`pdm run` → `uv run`),
  `scripts/cellclip/extract_missing_long.sh` (`.venv/bin/python` → `uv run python`),
  `.claude/settings.json` (ruff hook + permissions), `.gitignore`, `.vscode/settings.json`.
- Rewrote docs to a single uv + `uv run poe` install/run story; removed the "HPC Cluster Setup (Conda)" and
  dual PDM/HPC command paths.
