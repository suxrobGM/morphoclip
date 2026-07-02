# Project Structure Rules

## Source Layout

- MorphoCLIP library code lives under `src/morphoclip/`. Subpackages: `data`, `models`, `utils`.
- CellCLIP baseline code lives under `src/cellclip/`. Subpackages: `benchmark`, `training`. This is a separate package, not part of `morphoclip`.
- General benchmark code lives under `src/benchmark/`. Shared by both MorphoCLIP and CellCLIP.
- Each subpackage has an `__init__.py` that re-exports its public API via `__all__`.
- Do not add new top-level packages under `src/` without a clear domain boundary.

## Scripts

- Scripts live under `scripts/` organized by domain:
  - `data/`, `features/`, `text/`, `training/` — MorphoCLIP pipeline stages
  - `benchmark/` — general benchmark scripts (stable benchmark, comparison)
  - `cellclip/` — CellCLIP-specific scripts (training, export, pipeline)
- Every script starts with `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))` to resolve imports.
- Scripts are thin CLI wrappers — business logic belongs in `src/`.
- Task entries in `pyproject.toml` under `[tool.poe.tasks]` (run via `uv run poe <task>`).

## Tests

- Tests live under `tests/` mirroring `src/` structure: `tests/data/`, `tests/models/`, `tests/benchmark/`, `tests/cellclip/`.
- Each test subdirectory has an empty `__init__.py`.
- Shared fixtures go in `tests/conftest.py`.
- Tests that need local data files use `pytest.skip()` when data is absent.

## Configuration

- `configs/dataset.yml` — MorphoCLIP settings (S3 paths, plates, extraction, splits) under the `cpjump` key.
- `configs/train/` — MorphoCLIP training configs (`base.yaml`, `mean_pool.yaml`, `ddp.yaml`).
- `configs/benchmark.yml` — benchmark and CellCLIP export settings.
- `configs/cellclip/` — CellCLIP training configs with `extends` inheritance from `base.yaml`.

## Documentation

- `docs/` is a Nextra 4 project (Next.js 16.2 + Bun). It serves as the public docs website.
- Content lives in `docs/content/` as `.mdx` files organized by topic:
  - `getting-started/` — installation and quick start guides
  - `pipeline/` — training pipeline, feature extraction, text encoder, data fetching
  - `dataset/` — CPJUMP1 overview, splits, compression
  - `background/` — project proposal, literature review
  - `baselines/` — CellCLIP, benchmark guides and results
  - `glossary.mdx`, `team.mdx` — top-level pages
- Navigation defined via `_meta.ts` files (typed with `MetaRecord` from `nextra`) co-located with content.
- Internal-only docs (todo, plan, strategy) live in `docs/_internal/` and are not served.
- Next.js boilerplate (`app/`, `next.config.ts`, `package.json`) lives alongside content in `docs/`.
- Build: `cd docs && bun run build` produces static output in `docs/out/`.

## Baselines

- External comparison repos live as git submodules under `baselines/`.
- Current submodules: `baselines/CellCLIP`, `baselines/2024_Chandrasekaran_NatureMethods_CPJUMP1`.
- `baselines/cellprofiler/` contains the CellProfiler baseline notebook and utilities.
