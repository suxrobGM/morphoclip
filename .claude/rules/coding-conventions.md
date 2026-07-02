# Coding Conventions

## Python Style

- Target Python >=3.14. Use modern syntax: `type` unions (`X | Y`), `tuple[str, str]` not `Tuple`.
- Don't use `from __future__ import annotations` since Python 3.14 has PEP 563 semantics by default.
- Type hints on all public functions. Private helpers can omit them when types are obvious.
- Docstrings follow Google style with `Args:`, `Returns:`, `Raises:` sections.
- Use keyword-only arguments (`*`) for functions with more than 2-3 parameters, especially booleans.

## Imports

- Always use full package paths: `from morphoclip.data.metadata import MetadataIndex`, `from cellclip.training.config import CellCLIPTrainingConfig`.
- Never use relative imports within `src/` packages.
- Group imports: stdlib, third-party, local (`morphoclip.*`, `cellclip.*`, `benchmark.*`), separated by blank lines.
- CellCLIP code may import from `morphoclip.data` and `benchmark.data`, but `morphoclip` library code (`data`, `models`, `utils`) must never import from `cellclip`. The one exception is `morphoclip.cli`, the CLI composition root, which imports from `cellclip.*` and `benchmark.*` to expose their commands.
- In the remaining dev scripts under `scripts/`, place the `sys.path.insert` before local imports with `# noqa: E402` on the import lines. Package and CLI code never use `sys.path.insert`.

## Naming

- Modules: `snake_case.py`
- Classes: `PascalCase` (e.g., `MetadataIndex`, `PromptBuilder`, `ProjectionHead`)
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix with `_`

## File Size

- Target ~300–350 LOC per module. Split files that exceed ~350 lines.
- Extract dataclasses, constants, and pure functions into their own modules first.
- Keep orchestration logic in the original module.

## Comments and Docstrings

- Keep docstrings concise. Include `Args:`/`Returns:` but don't over-explain obvious parameters.
- Docstrings can be skipped only for trivially obvious private helpers (e.g., simple getters).
- No inline comments that restate the code. Only comment the *why*, not the *what*.

## Error Handling

- Raise `ValueError` for invalid arguments.
- Raise `RuntimeError` for environment/system issues (missing tools, failed commands).
- Use `subprocess.run(cmd, check=True)` — let `CalledProcessError` propagate.

## Testing

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>` for grouped tests, plain functions for standalone tests.
- Use `tmp_path` fixture for filesystem tests.
- Use `pytest.mark.skipif` for tests that need optional dependencies or local data.
- Run tests: `uv run poe test` (which runs `pytest tests/ -v`).
