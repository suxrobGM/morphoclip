# Project structure

## Source

```text
src/
  morphoclip/
    config.py      pydantic StrictModel base, `extends` loader, --set overrides
    cli/           the `morphoclip` Typer app
    data/          metadata, dataset, image loading, extraction pipeline
    models/        text and image encoders, prompts
    splits/        dataset splitting
    training/      trainer, losses, optim, inference, retrieval
    utils/         caching, console, device, s3, hf transfer
    benchmark/     profile export for MorphoCLIP checkpoints
  benchmark/       CPJUMP1 evaluation. Standalone: no torch, no morphoclip.
  cellclip/        frozen baseline (benchmark/ and training/ subpackages)
```

The project is an installable package (hatchling, `[tool.uv] package = true`).
Never reintroduce `sys.path.insert` in package or CLI code.

`__init__.py` files are empty. Re-exporting through them pulled TensorBoard,
transformers and pandas into every import; `tests/test_import_graph.py` budgets
the graph so that cannot come back.

## CLI

One Typer app, exposed as the `morphoclip` console script and runnable as
`python -m morphoclip.cli` (which is how `torchrun` launches it).

One module per command group under `src/morphoclip/cli/`: top-level `train`,
`eval`, `infer`, `split`, `benchmark`, `export-profiles`, plus the `data`,
`features`, `text` and `cellclip` sub-apps.

- Command bodies are thin. Logic belongs in `morphoclip.*`, `cellclip.*` or
  `benchmark.*`.
- Imports that pull an optional extra (`benchmark.stable` needs copairs and
  scikit-learn) must be **lazy**, inside the command body, so
  `morphoclip --help` works without the extra installed.
- Repeated Typer options are shared `Annotated` aliases in `cli/options.py`.
  Use plain module-level assignment, not PEP 695 `type X = ...`: Typer cannot
  see through a lazily-evaluated `TypeAliasType`.

## Scripts

`scripts/` holds dev and exploration one-offs only, organised by domain
(`data/`, `features/`, `text/`, `benchmark/`, `sanitycheck/`). They are not part
of the installed package and start with
`sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))`.

Do not add a pipeline entry point here. Add a `morphoclip.cli` command.

## Tests

`tests/` mirrors `src/morphoclip/`: `data/`, `models/`, `splits/`, `training/`,
`config/`, `cli/`. Each subdirectory has an empty `__init__.py`.

There is deliberately no test suite for `benchmark` or `cellclip`. Both are
frozen, their results are captured in `report/`, and a suite for them was not
worth maintaining.

- Shared builders live in `tests/support/` and are imported as
  `from tests.support.features import ...`. Never import from a conftest.
- `tests/conftest.py` provides session-scoped `metadata_dir` and
  `metadata_index` over the committed fixture in `tests/fixtures/cpjump1/`.
  Tests must not read `data/`, which is gitignored and absent in CI.
- An autouse fixture raises on `socket.connect`, so a test that forgets to
  patch `from_pretrained` fails instead of downloading.
- `filterwarnings = ["error"]`. A warning fails the test.
- Tests needing the downloaded dataset are marked `realdata` and excluded from
  the default run.

The bar for a test: name a plausible one-line source change that makes it fail
and that produces a wrong result. A test that can only fail on a rename is
churn, and gets deleted.

## Configuration

- `configs/dataset.yml` under the `cpjump` key: S3 paths, plates, extraction,
  splits. Parsed only by `morphoclip.data.config`.
- `configs/train/` MorphoCLIP training configs.
- `configs/benchmark.yml` benchmark and CellCLIP export settings.
- `configs/cellclip/` five configs with `extends` inheritance. Historical
  variants are `--set` commands in `configs/cellclip/README.md`.

Every committed config has a resolution golden in `tests/config/goldens/`.

## Tasks

`[tool.poe.tasks]` holds the dev loop (`test`, `lint`, `format`, `typecheck`,
`check`), the two commands that supply a default config (`train`, `benchmark`),
`tensorboard`, and the script runners. Pipeline commands are **not** aliased
there: a second name for each one only gave the docs something to drift from.

## Documentation

`docs/` is a Nextra 4 site (Next.js, Bun). Content is `.mdx` under
`docs/content/`, navigation is `_meta.ts` files co-located with it. Internal
notes live in `docs/_internal/` and are not served. Build with
`cd docs && bun run build`.
