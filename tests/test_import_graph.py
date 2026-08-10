"""Budget the import graph so a stray top-level import cannot creep back.

Emptying the package __init__ files took `import morphoclip.training.config`
from 3,918 modules to about 200. That is easy to undo by accident: one
convenience re-export in an __init__, or one torch import moved to module scope,
puts it back. Each check runs in a subprocess so sys.modules starts clean.
"""

import subprocess
import sys

import pytest


def _probe(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.parametrize(
    ("module", "budget", "forbidden"),
    [
        # Pure dataclasses and YAML. Used to pull TensorBoard, transformers and pandas.
        ("morphoclip.training.config", 400, "torch"),
        # Pandas only. profile_ops exists so result accumulation is importable
        # without copairs, which benchmark.metrics needs.
        ("benchmark.data", 900, "benchmark.metrics"),
    ],
)
def test_module_import_stays_lean(module: str, budget: int, forbidden: str) -> None:
    count, pulled = _probe(
        f"import sys, {module}; print(len(sys.modules), {forbidden!r} in sys.modules)"
    ).split()
    assert int(count) < budget, (
        f"importing {module} loaded {count} modules (budget {budget}). "
        "Something gained a heavy top-level import."
    )
    assert pulled == "False", f"{module} pulled in {forbidden}"


@pytest.mark.parametrize("forbidden", ["torch", "morphoclip"])
def test_the_whole_benchmark_package_stays_free_of(forbidden: str) -> None:
    """`benchmark` is standalone. It used to import a torch Dataset through splits."""
    loaded = _probe(
        "import importlib, pkgutil, sys, benchmark\n"
        "for m in [x.name for x in pkgutil.iter_modules(benchmark.__path__)]:\n"
        "    importlib.import_module('benchmark.' + m)\n"
        f"print(any(k == {forbidden!r} or k.startswith({forbidden!r} + '.') for k in sys.modules))"
    )
    assert loaded == "False", f"a benchmark module now imports {forbidden}"
