---
paths:
  - "**/*.py"
---

# Python

- Target Python 3.14: `X | Y`, `list[str]`, PEP 695 generics. No
  `Optional`/`List`/`Tuple`, no `from __future__ import annotations`.
- Type-hint every public function.
- Make parameters keyword-only (`*`) past the second or third, and always for
  booleans.
- Order imports stdlib, third-party, first-party (`morphoclip`, `cellclip`,
  `benchmark`).
- Import optional extras (copairs, scikit-learn) inside the function that uses them.
- Use `sys.path.insert` only in `scripts/`.
- Naming: `snake_case` modules and functions, `PascalCase` classes,
  `UPPER_SNAKE_CASE` constants, `_prefixed` private helpers.
- Never reuse an existing public name for a different thing.
- Write Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
- Comment only the why. No banner comments, no numbered step comments.
- Never use em or en dashes, including in docstrings.
- Do not add comments to lines you are not otherwise changing.
- Raise `ValueError` for bad arguments, `RuntimeError` for a broken environment.
- Call `subprocess.run(..., check=True)` and let `CalledProcessError` propagate.
- Never widen an `except` clause to silence a failure.
- Split a file when it does two unrelated jobs, not because of length. Move
  dataclasses, constants and pure functions out; keep orchestration in place.
