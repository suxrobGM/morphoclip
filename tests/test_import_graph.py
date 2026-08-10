"""Budget the import graph so a stray top-level import cannot creep back.

Emptying the package __init__ files took `import morphoclip.training.config`
from 3,918 modules to about 130. That is easy to undo by accident: one
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
    ("module", "budget"),
    [
        # Pure dataclasses and YAML. Used to pull TensorBoard, transformers and pandas.
        ("morphoclip.training.config", 400),
        # Pandas only. Used to pull benchmark.metrics through the package __init__.
        ("benchmark.data", 900),
    ],
)
def test_module_import_stays_within_budget(module: str, budget: int) -> None:
    loaded = int(_probe(f"import sys, {module}; print(len(sys.modules))"))
    assert loaded < budget, (
        f"importing {module} loaded {loaded} modules (budget {budget}). "
        "Something gained a heavy top-level import."
    )


def test_the_training_config_does_not_import_torch() -> None:
    """`benchmark` gets the same guarantee from the whole-package test below."""
    loaded = _probe("import sys, morphoclip.training.config; print('torch' in sys.modules)")
    assert loaded == "False", "morphoclip.training.config pulled in torch"


def test_benchmark_data_does_not_pull_the_metrics_stack() -> None:
    """profile_ops exists so result accumulation is importable without copairs."""
    loaded = _probe("import sys, benchmark.data; print('benchmark.metrics' in sys.modules)")
    assert loaded == "False"


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
