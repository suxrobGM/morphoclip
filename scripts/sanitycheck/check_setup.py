#!/usr/bin/env python3
"""Basic environment sanity check for MorphoCLIP setups."""

import importlib
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()

# Body messages carry literal ``[OK]``/``[FAIL]`` tokens, so they are printed with
# ``markup=False`` and colorized via ``style=`` to avoid Rich parsing the brackets.
_STATUS_STYLES = {"[OK]": "green", "[FAIL]": "red bold", "[WARN]": "yellow", "[INFO]": "cyan"}

REQUIRED_IMPORTS = [
    "PIL",
    "numpy",
    "yaml",
    "cv2",
    "transformers",
    "huggingface_hub",
    "rich",
    "pandas",
    "scipy",
    "skimage",
    "dotenv",
    "torch",
    "torchvision",
    "matplotlib",
    "seaborn",
    "datasets",
]

OPTIONAL_IMPORTS = [
    "tensorboard",
]


class DeviceMode(StrEnum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


def _emit(message: str) -> None:
    style = next((s for prefix, s in _STATUS_STYLES.items() if message.startswith(prefix)), None)
    console.print(f"  - {message}", style=style, markup=False)


def _check_python_version() -> tuple[bool, str]:
    version = sys.version_info
    ok = (version.major, version.minor) == (3, 14)
    msg = f"Python version: {version.major}.{version.minor}.{version.micro}"
    if not ok:
        msg += " (expected 3.14.x)"
    return ok, msg


def _check_imports() -> tuple[bool, list[str]]:
    messages: list[str] = []
    all_ok = True
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            messages.append(f"[OK] import {module_name}")
        except Exception as exc:  # pragma: no cover - defensive path
            all_ok = False
            messages.append(f"[FAIL] import {module_name}: {exc}")
    return all_ok, messages


def _check_optional_imports() -> list[str]:
    messages: list[str] = []
    for module_name in OPTIONAL_IMPORTS:
        try:
            importlib.import_module(module_name)
            messages.append(f"[OK] import {module_name}")
        except Exception:
            messages.append(f"[WARN] import {module_name}: not installed (optional)")
    return messages


def _check_project_files() -> tuple[bool, list[str]]:
    repo_root = Path(__file__).resolve().parents[2]
    required_paths = [
        repo_root / "configs" / "dataset.yml",
        repo_root / "configs" / "benchmark.yml",
        repo_root / "src" / "morphoclip",
        repo_root / "scripts",
    ]
    messages: list[str] = []
    all_ok = True
    for path in required_paths:
        if path.exists():
            messages.append(f"[OK] found {path.relative_to(repo_root)}")
        else:
            all_ok = False
            messages.append(f"[FAIL] missing {path.relative_to(repo_root)}")
    return all_ok, messages


def _check_torch(device_mode: str, expected_cuda: str | None) -> tuple[bool, list[str]]:
    messages: list[str] = []
    try:
        import torch
    except Exception as exc:
        return False, [f"[FAIL] import torch: {exc}"]

    ok = True
    messages.append(f"[OK] torch version: {torch.__version__}")
    messages.append(f"[OK] torch CUDA runtime: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    messages.append(f"[OK] torch.cuda.is_available(): {cuda_available}")

    if device_mode == "gpu":
        if not cuda_available:
            return False, messages + ["[FAIL] GPU required but CUDA is unavailable."]
        device = "cuda"
    elif device_mode == "cpu":
        device = "cpu"
    else:
        device = "cuda" if cuda_available else "cpu"

    if expected_cuda:
        actual = torch.version.cuda or ""
        if not actual.startswith(expected_cuda):
            ok = False
            messages.append(
                "[FAIL] expected CUDA runtime to start with "
                f"'{expected_cuda}', got '{actual or 'None'}'"
            )
        else:
            messages.append(f"[OK] expected CUDA runtime '{expected_cuda}' matched")

    try:
        x = torch.randn(256, 256, device=device)
        y = torch.randn(256, 256, device=device)
        z = x @ y
        _ = z.norm().item()
        messages.append(f"[OK] tensor matmul/norm on device: {device}")
    except Exception as exc:
        ok = False
        messages.append(f"[FAIL] tensor operation on device '{device}': {exc}")

    return ok, messages


def main(
    device: Annotated[
        DeviceMode, typer.Option(help="Device mode for torch sanity check.")
    ] = DeviceMode.auto,
    expected_cuda: Annotated[
        str | None, typer.Option(help="Expected CUDA runtime prefix, e.g. '12.1', '12.4', '12.8'.")
    ] = None,
) -> None:
    """Sanity check MorphoCLIP environment."""
    checks: list[tuple[str, bool, list[str]]] = []

    py_ok, py_msg = _check_python_version()
    checks.append(("Python", py_ok, [py_msg]))

    imports_ok, import_msgs = _check_imports()
    checks.append(("Core imports", imports_ok, import_msgs))

    optional_msgs = _check_optional_imports()
    checks.append(("Optional imports", True, optional_msgs))

    files_ok, file_msgs = _check_project_files()
    checks.append(("Project layout", files_ok, file_msgs))

    torch_ok, torch_msgs = _check_torch(device, expected_cuda)
    checks.append(("Torch/CUDA", torch_ok, torch_msgs))

    console.rule("[bold blue]MorphoCLIP Environment Sanity Check")
    overall_ok = True
    for name, ok, messages in checks:
        status = "PASS" if ok else "FAIL"
        console.print(f"[{status}] {name}", style="green bold" if ok else "red bold", markup=False)
        for message in messages:
            _emit(message)
        console.print()
        overall_ok = overall_ok and ok

    if overall_ok:
        console.print("[green]All sanity checks passed.[/green]")
        return

    console.print("[red]One or more sanity checks failed.[/red]")
    raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
