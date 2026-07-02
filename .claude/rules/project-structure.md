# Project Structure Rules

## Source Layout

- MorphoCLIP library code lives under `src/morphoclip/`. Subpackages: `cli`, `data`, `models`, `utils`.
- CellCLIP baseline code lives under `src/cellclip/`. Subpackages: `benchmark`, `training`. This is a separate package, not part of `morphoclip`.
- General benchmark code lives under `src/benchmark/`. Shared by both MorphoCLIP and CellCLIP.
- Each subpackage has an `__init__.py` that re-exports its public API via `__all__`.
- The project is an installable package (hatchling build backend, `[tool.uv] package = true`). Do not reintroduce `sys.path.insert` hacks in package or CLI code.
- Do not add new top-level packages under `src/` without a clear domain boundary.

## CLI (`morphoclip.cli`)

- Pipeline entry points are a single Typer app exposed as the `morphoclip` console command (`[project.scripts]`), also runnable via `python -m morphoclip.cli` (used by `torchrun`).
- One module per command group under `src/morphoclip/cli/`: top-level commands (`train`, `eval`, `infer`, `split`, `benchmark`) plus sub-apps `data`, `features`, `text`, `cellclip`.
- CLI command bodies are thin wrappers — business logic belongs in `src/` (`morphoclip.*`, `cellclip.*`, `benchmark.*`). Command-specific orchestration may live in the command module; large/reusable logic (e.g. the stable benchmark) lives in its package (`benchmark.stable`).
- Imports that pull optional extras (e.g. `benchmark.stable` → copairs/sklearn) must be **lazy** (inside the command body) so `morphoclip --help` works without those extras installed.
- Poe tasks in `[tool.poe.tasks]` wrap CLI commands (e.g. `train = "morphoclip train ..."`); run via `uv run poe <task>` or `uv run morphoclip <command>`.

## Scripts

- `scripts/` holds only dev/exploration one-offs (inspection, diagnostics, analysis, sanity checks), organized by domain (`data/`, `features/`, `text/`, `benchmark/`, `cellclip/`, `analysis/`, `sanitycheck/`).
- These scripts start with `sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))` to resolve imports (they are not part of the installed package).
- Do not add new pipeline entry points here — add a `morphoclip.cli` command instead.

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

- External comparison repos are **not** vendored. Reference them upstream by link:
  [CellCLIP](https://github.com/suinleelab/CellCLIP) and
  [Chandrasekaran 2024 CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1).
- The small CPJUMP1 reference metadata required by the benchmark harness lives
  first-party under `data/reference/cpjump1/` (`cpjump1_metadata.csv`,
  `JUMP-Target-1_compound_metadata_additional_annotations.tsv`).
