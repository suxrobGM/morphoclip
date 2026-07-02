"""Centralized device selection and mixed-precision helpers.

Provides a single ``resolve_device`` function that every script and
training loop should use so the priority order (CUDA > MPS > CPU) is
consistent across the codebase.
"""

import os
from contextlib import nullcontext

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    When *device* is ``"auto"`` the best available accelerator is
    selected with priority **CUDA > MPS > CPU**.

    Args:
        device: ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``, or any
            string accepted by :class:`torch.device`.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def autocast_context(device: torch.device, enabled: bool):
    """Create an autocast context for mixed-precision forward passes.

    Supports CUDA and MPS (PyTorch >= 2.1).  Returns a
    :func:`~contextlib.nullcontext` for CPU or when *enabled* is False.
    """
    if not enabled or device.type not in ("cuda", "mps"):
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def build_grad_scaler(device: torch.device, *, enabled: bool) -> torch.amp.GradScaler:
    """Build a :class:`~torch.amp.GradScaler` appropriate for *device*."""
    if device.type == "cuda" and enabled:
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)


def supports_pin_memory(device: torch.device) -> bool:
    """Return True if pinned memory benefits data loading for *device*."""
    return device.type == "cuda"


def resolve_num_workers(requested: int) -> int:
    """Clamp DataLoader workers to available CPU cores.

    Uses ``os.sched_getaffinity`` (respects SLURM/cgroup limits) with
    a fallback to ``os.cpu_count``.  Reserves 1 core for the main
    process.

    Args:
        requested: Number of workers from config.

    Returns:
        ``min(requested, available_cores - 1)``, at least 0.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    max_workers = max(0, available - 1)
    return min(requested, max_workers)
