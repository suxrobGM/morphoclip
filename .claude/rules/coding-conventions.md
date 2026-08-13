# Coding conventions

## Python style

- Target Python >=3.14. Use modern syntax: `X | Y` unions, `tuple[str, str]`
  instead of `Tuple`.
- No `from __future__ import annotations`. Python 3.14 already behaves that way.
- Type hints on all public functions. Private helpers may skip them when the
  types are obvious.
- Google-style docstrings with `Args:`, `Returns:`, `Raises:`.
- Make arguments keyword-only (`*`) past two or three parameters, especially
  booleans.
- PEP 695 generics are fine (`def load_config[ModelT: BaseModel]`). The one
  exception is Typer option aliases, which must be plain assignment.

## Imports

- Full package paths: `from morphoclip.data.metadata import MetadataIndex`.
- Never use relative imports inside `src/`.
- Group imports: stdlib, third-party, local (`morphoclip.*`, `cellclip.*`,
  `benchmark.*`).
- `morphoclip` library code must never import from `cellclip`. The one
  exception is `morphoclip.cli`, which wires everything together.
- An import that pulls in an optional extra goes inside the function that
  needs it.
- Only `scripts/` may use `sys.path.insert`. It goes before the local imports,
  with `# noqa: E402`.

## Naming

- Modules `snake_case.py`, classes `PascalCase`, functions `snake_case`,
  constants `UPPER_SNAKE_CASE`, private helpers `_prefixed`.
- Give each thing one name. This codebase once had three importable things
  called `benchmark` and two called `setup_logging`, and that is exactly how
  it got confusing.

## File size

Aim for 300 to 350 lines. Split a file when it does two unrelated jobs, not
because it crossed a line count. Splitting a coherent module by size just
produces two files you have to read together anyway.

When splitting, move out dataclasses, constants and pure functions first.
Keep the orchestration in the original module.

## Comments and docstrings

- Comment the why, never the what. Default to no comment.
- No banner comments (`# --- Helpers ---`) and no numbered step comments.
- No em dashes or en dashes anywhere, including docstrings. Use a comma,
  colon, period or parentheses.
- Do not add comments to lines you did not otherwise change.

## Error handling

- `ValueError` for a bad argument, `RuntimeError` for a broken environment.
- `subprocess.run(cmd, check=True)`, and let `CalledProcessError` propagate.
- Never widen an `except` clause to silence a failure. The unpaired guard in
  `benchmark` is deliberately narrow: widening it would have turned a crash
  into published zeros.

## Testing

- Files are `test_<module>.py`, mirroring `src/morphoclip/`.
- Use `Test<Thing>` classes to group related tests, plain functions otherwise.
- A test name states the contract it checks:
  `test_dry_run_reports_the_count_without_deleting`.
- Use `tmp_path` and the builders in `tests/support/`. Do not hand-build a
  fake feature root, plate image directory or profile CSV when a builder
  already exists.
- Prefer one `parametrize` over several tests that differ only by input.
- Mark tests that need the downloaded dataset with `realdata`. Do not add a
  `skipif` that quietly passes on a fresh clone.
- Before adding a test, name the one-line source change it would catch. If the
  only thing that can make it fail is a rename, do not write it.

Run with `uv run poe test`, or `uv run poe check` for format, lint, types and
tests together.
