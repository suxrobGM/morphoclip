#!/usr/bin/env python3
"""GPU sanity checks for MorphoCLIP environments."""

import os

# macOS: PyTorch and other libs may each load libomp; without this, Intel OMP aborts with OMP #15.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import platform
import sys
from dataclasses import dataclass


@dataclass
class DeviceStats:
    index: int
    name: str
    capability: str
    total_gb: float
    free_gb: float
    used_gb: float
    usable_estimate_gb: float


def _to_gb(bytes_value: int) -> float:
    return bytes_value / (1024**3)


def _try_allocate_mib(torch_module, device: int, size_mib: int) -> bool:
    """Try allocating a tensor of size_mib on one GPU."""
    if size_mib <= 0:
        return True
    bytes_needed = size_mib * 1024 * 1024
    elems = bytes_needed // 4  # float32 bytes
    try:
        tensor = torch_module.empty(elems, dtype=torch_module.float32, device=f"cuda:{device}")
        del tensor
        torch_module.cuda.empty_cache()
        return True
    except RuntimeError:
        torch_module.cuda.empty_cache()
        return False


def _probe_max_alloc_mib(torch_module, device: int, upper_bound_mib: int) -> int:
    """Binary search max allocatable contiguous VRAM in MiB."""
    low = 0
    high = max(0, upper_bound_mib)
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if _try_allocate_mib(torch_module, device, mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GPU availability and VRAM statistics.")
    parser.add_argument(
        "--probe-max-alloc",
        action="store_true",
        help="Empirically probe maximum allocatable contiguous VRAM per GPU (CUDA only).",
    )
    parser.add_argument(
        "--probe-fraction",
        type=float,
        default=0.90,
        help=(
            "Upper bound fraction of currently free VRAM for max-allocation probe (default: 0.90)."
        ),
    )
    return parser.parse_args()


def _run_cuda_check(torch_module, args: argparse.Namespace) -> int:
    device_count = torch_module.cuda.device_count()
    print("Backend: CUDA")
    print(f"Detected GPUs: {device_count}")
    if device_count > 1:
        print("[OK] Multi-GPU environment detected.")
    else:
        print("[INFO] Single GPU detected.")
    print()

    stats: list[DeviceStats] = []
    for idx in range(device_count):
        props = torch_module.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch_module.cuda.mem_get_info(idx)
        used_bytes = total_bytes - free_bytes
        stats.append(
            DeviceStats(
                index=idx,
                name=props.name,
                capability=f"{props.major}.{props.minor}",
                total_gb=_to_gb(total_bytes),
                free_gb=_to_gb(free_bytes),
                used_gb=_to_gb(used_bytes),
                usable_estimate_gb=_to_gb(int(free_bytes * 0.9)),
            )
        )

    for item in stats:
        print(f"[GPU {item.index}] {item.name}")
        print(f"  - compute capability: {item.capability}")
        print(f"  - total VRAM: {item.total_gb:.2f} GB")
        print(f"  - free VRAM now: {item.free_gb:.2f} GB")
        print(f"  - used VRAM now: {item.used_gb:.2f} GB")
        print(f"  - estimated usable now (~90% free): {item.usable_estimate_gb:.2f} GB")

        if args.probe_max_alloc:
            upper_mib = int((item.free_gb * 1024) * max(0.10, min(args.probe_fraction, 0.98)))
            max_mib = _probe_max_alloc_mib(torch_module, item.index, upper_mib)
            print(f"  - probed max contiguous alloc: {max_mib / 1024:.2f} GB")
        print()

    print("GPU sanity check complete.")
    return 0


def _run_mps_check(torch_module, args: argparse.Namespace) -> int:
    """Apple Silicon Metal (MPS) path when CUDA is unavailable."""
    print("Backend: MPS (Metal)")
    print(f"MPS built: {torch_module.backends.mps.is_built()}")
    print(f"MPS available: {torch_module.backends.mps.is_available()}")
    if not torch_module.backends.mps.is_available():
        print("[FAIL] MPS is not available (needs Apple Silicon and a supported macOS).")
        return 1

    try:
        x = torch_module.zeros(4, 4, device="mps")
        del x
    except Exception as exc:
        print(f"[FAIL] Could not allocate on MPS device: {exc}")
        return 1

    print("[OK] Metal (MPS) device accepts tensors.")
    if hasattr(torch_module.mps, "current_allocated_memory"):
        cur = torch_module.mps.current_allocated_memory()
        print(f"  - MPS current allocated (reported): {_to_gb(cur):.4f} GB")
    if hasattr(torch_module.mps, "driver_allocated_memory"):
        drv = torch_module.mps.driver_allocated_memory()
        print(f"  - MPS driver allocated (reported): {_to_gb(drv):.2f} GB")

    if args.probe_max_alloc:
        print("[INFO] --probe-max-alloc is only implemented for CUDA; ignored on MPS.")

    print()
    print("GPU sanity check complete.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        import torch
    except Exception as exc:
        print(f"[FAIL] Could not import torch: {exc}")
        return 1

    print("=== MorphoCLIP GPU Sanity Check ===")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA runtime (torch): {torch.version.cuda}")
    print()

    if torch.cuda.is_available():
        return _run_cuda_check(torch, args)

    if torch.backends.mps.is_available():
        return _run_mps_check(torch, args)

    print(
        "[FAIL] No GPU backend available: CUDA is not available and MPS (Metal) is not available."
    )
    print("        On Apple Silicon Macs, install a PyTorch build with MPS support.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
