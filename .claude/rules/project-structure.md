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
Never bring `sys.path.insert` back into package or CLI code.

All `__init__.py` files are empty. They used to re-export things, which pulled
TensorBoard, transformers and pandas into every import.
`tests/test_import_graph.py` puts a budget on the import graph so that cannot
happen again.

## CLI

One Typer app, installed as the `morphoclip` console script. It can also run
as `python -m morphoclip.cli`, which is how `torchrun` launches it.

Each command group is one module under `src/morphoclip/cli/`: the top-level
commands `train`, `eval`, `infer`, `split`, `benchmark`, `export-profiles`,
plus the `data`, `features`, `text` and `cellclip` sub-apps.

- Command bodies are thin. Logic belongs in `morphoclip.*`, `cellclip.*` or
  `benchmark.*`.
- An import that pulls in an optional extra (`benchmark.stable` needs copairs
  and scikit-learn) must be lazy, inside the command body. Otherwise
  `morphoclip --help` breaks when the extra is not installed.
- Typer options used by several commands are shared `Annotated` aliases in
  `cli/options.py`. Define them with plain module-level assignment, not
  PEP 695 `type X = ...`: Typer cannot read a lazily-evaluated
  `TypeAliasType`.

## Scripts

`scripts/` holds one-off dev and exploration scripts, organised by domain
(`data/`, `features/`, `text/`, `benchmark/`, `sanitycheck/`). They are not
part of the installed package, so each one starts with
`sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))`.

Do not add a pipeline entry point here. Add a `morphoclip.cli` command
instead.

## Tests

`tests/` mirrors `src/morphoclip/`: `data/`, `models/`, `splits/`,
`training/`, `config/`, `cli/`. Each subdirectory has an empty `__init__.py`.

There is no test suite for `benchmark` or `cellclip`, on purpose. Both are
frozen, their results are recorded in `report/`, and maintaining a suite for
them was not worth it.

- Shared builders live in `tests/support/` and are imported as
  `from tests.support.features import ...`. Never import from a conftest.
- `tests/conftest.py` provides session-scoped `metadata_dir` and
  `metadata_index` fixtures over the committed fixture data in
  `tests/fixtures/cpjump1/`. Tests must not read `data/`: it is gitignored
  and does not exist in CI.
- An autouse fixture raises on `socket.connect`. A test that forgets to patch
  `from_pretrained` fails instead of downloading a model.
- `filterwarnings = ["error"]`, so any warning fails the test.
- Tests that need the downloaded dataset are marked `realdata` and excluded
  from the default run.

The bar for a test: name a plausible one-line source change that makes it
fail and that would produce a wrong result. A test that can only fail on a
rename is churn, and gets deleted.

## Configuration

- `configs/dataset.yml`, under the `cpjump` key: S3 paths, plates, extraction,
  splits. Parsed only by `morphoclip.data.config`.
- `configs/train/`: MorphoCLIP training configs.
- `configs/benchmark.yml`: benchmark and CellCLIP export settings.
- `configs/cellclip/`: five configs linked by `extends` inheritance.
  Historical variants are recorded as `--set` commands in
  `configs/cellclip/README.md`.

Every committed config has a resolution golden in `tests/config/goldens/`.

## Tasks

`[tool.poe.tasks]` holds the dev loop (`test`, `lint`, `format`, `typecheck`,
`check`), the two commands that supply a default config (`train`,
`benchmark`), `tensorboard`, and the script runners. Pipeline commands are
deliberately not aliased there: a second name for each command only gave the
docs something to drift out of sync with.

## Documentation

`docs/` is a Nextra 4 site (Next.js, Bun). Content is `.mdx` under
`docs/content/`, and navigation is the `_meta.ts` files next to it. Internal
notes live in `docs/_internal/` and are not served. Build with
`cd docs && bun run build`.
