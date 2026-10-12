---
paths:
  - "tests/**"
---

# Tests

- Before adding a test, name the one-line source change that would make it
  fail with a wrong result. If only a rename can break it, do not write it.
- Mirror `src/morphoclip/` in `tests/`. Name files `test_<module>.py`.
- Do not add tests for `src/benchmark/` or `src/cellclip/`.
- Group related tests in `Test<Thing>` classes. Name each test after its
  contract: `test_dry_run_reports_the_count_without_deleting`.
- Use one `parametrize` instead of several tests that differ only by input.
- Use `tmp_path` and the builders in `tests/support/`. Never hand-build a
  feature root, image directory or profile CSV that a builder already makes.
- Import helpers from `tests.support.*`, never from a conftest.
- Use the `metadata_dir` / `metadata_index` fixtures. Never read `data/`.
- Patch every `from_pretrained` call (stubs in `tests/support/stubs.py`).
  Network access raises.
- Mark tests that need the downloaded dataset `realdata`. Never use a `skipif`
  that passes silently on a fresh clone.
- Mark tests that take more than a few seconds `slow`.
