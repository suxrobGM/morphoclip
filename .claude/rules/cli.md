---
paths:
  - "src/morphoclip/cli/**"
  - "scripts/**"
  - "pyproject.toml"
---

# CLI and scripts

- Add every pipeline step as a `morphoclip` command, never as a script.
- Keep command bodies thin. Put logic in `morphoclip.*`, `cellclip.*` or `benchmark.*`.
- Register new commands and sub-apps in `cli/__init__.py`.
- Import optional extras (`benchmark.stable`, copairs, scikit-learn) inside
  the command body, so `morphoclip --help` works without them.
- Define shared Typer `Annotated` aliases with plain assignment, never PEP 695
  `type X = ...`. Typer cannot read a `TypeAliasType`.
- Do not add poe aliases for pipeline commands.
- Put scripts in a domain folder under `scripts/`. A script that imports from
  `src/` inserts `Path(__file__).resolve().parents[2] / "src"` into
  `sys.path` before local imports and marks them `# noqa: E402`.
