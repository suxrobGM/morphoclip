# Coding conventions

## Python style

- Target Python >=3.14. Modern syntax: `X | Y` unions, `tuple[str, str]` not `Tuple`.
- No `from __future__ import annotations`. Python 3.14 already has those semantics.
- Type hints on all public functions. Private helpers may omit them when obvious.
- Google-style docstrings with `Args:`, `Returns:`, `Raises:`.
- Keyword-only arguments (`*`) past two or three parameters, especially booleans.
- PEP 695 generics are fine (`def load_config[ModelT: BaseModel]`), except for
  Typer option aliases, which must be plain assignment.

## Imports

- Full package paths: `from morphoclip.data.metadata import MetadataIndex`.
- Never relative imports inside `src/`.
- Group: stdlib, third-party, local (`morphoclip.*`, `cellclip.*`, `benchmark.*`).
- `morphoclip` library code must never import from `cellclip`. `morphoclip.cli`
  is the one exception; it is the composition root.
- Imports that pull an optional extra go inside the function that needs them.
- Only `scripts/` uses `sys.path.insert`, before its local imports, with
  `# noqa: E402`.

## Naming

- Modules `snake_case.py`, classes `PascalCase`, functions `snake_case`,
  constants `UPPER_SNAKE_CASE`, private helpers `_prefixed`.
- Name a thing once. Three importable things called `benchmark` and two called
  `setup_logging` is how this codebase got confusing the first time.

## File size

Aim for 300 to 350 lines. Split a file when it does two unrelated jobs, not
because it crossed a line count: splitting a coherent module by size produces
two files that must be read together.

Extract dataclasses, constants and pure functions first. Keep orchestration in
the original module.

## Comments and docstrings

- Comment the why, never the what. Default to no comment.
- No banner comments (`# --- Helpers ---`) and no numbered step comments.
- No em dashes or en dashes anywhere, including docstrings. Use a comma, colon,
  period or parentheses.
- Do not add comments to lines you did not otherwise change.

## Error handling

- `ValueError` for a bad argument, `RuntimeError` for a broken environment.
- `subprocess.run(cmd, check=True)`; let `CalledProcessError` propagate.
- Do not widen an `except` clause to make a failure quiet. `benchmark`'s
  unpaired guard is deliberately narrow because widening it would have turned a
  crash into published zeros.

## Testing

- Files `test_<module>.py`, mirroring `src/morphoclip/`.
- `Test<Thing>` classes for grouped tests, plain functions otherwise.
- Test names are contract sentences: `test_dry_run_reports_the_count_without_deleting`.
- Use `tmp_path` and the builders in `tests/support/`. Do not hand-roll a fake
  feature root, plate image directory or profile CSV that already has a builder.
- Prefer one `parametrize` over several tests that differ only by input.
- Mark `realdata` for anything needing the downloaded dataset. Do not add a
  `skipif` that silently passes on a fresh clone.
- Before adding a test, name the one-line source change it catches. If the only
  way it can fail is a rename, do not write it.

Run with `uv run poe test`, or `uv run poe check` for format, lint, types and
tests together.
