"""Distributed training utilities for multi-GPU DDP.

Encapsulates all ``torch.distributed`` concerns so that the rest of the
codebase can remain largely unaware of DDP.  When launched without
``torchrun`` (no ``RANK`` in env), everything degrades gracefully to
single-process mode.

Usage with torchrun::

    torchrun --nproc_per_node=4 scripts/training/train.py \\
        --config configs/train/ddp.yaml --distributed
"""

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

# ------------------------------------------------------------------
# Distributed state
# ------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DistributedState:
    """Immutable snapshot of the current distributed process group.

    Attributes:
        rank: Global rank of this process.
        local_rank: Local rank (GPU index on this node).
        world_size: Total number of processes.
        is_main: ``True`` only on rank 0.
        backend: Communication backend (e.g. ``"nccl"``).
        device: ``torch.device`` assigned to this process.
    """

    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    backend: str
    device: torch.device


def setup_distributed(backend: str = "nccl") -> DistributedState:
    """Initialize the distributed process group.

    Reads ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE`` from the
    environment (set automatically by ``torchrun``).  If these are
    absent, returns a single-process ``DistributedState`` using
    :func:`~morphoclip.utils.device.resolve_device`.

    Args:
        backend: Communication backend (``"nccl"`` for GPU).

    Returns:
        Populated ``DistributedState``.
    """
    if "RANK" not in os.environ:
        from morphoclip.utils.device import resolve_device

        device = resolve_device("auto")
        return DistributedState(
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            backend=backend,
            device=device,
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)

    device = torch.device("cuda", local_rank)
    return DistributedState(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=(rank == 0),
        backend=backend,
        device=device,
    )


def cleanup_distributed() -> None:
    """Destroy the distributed process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Return ``True`` if a distributed process group is active."""
    return dist.is_initialized()


# ------------------------------------------------------------------
# Communication helpers
# ------------------------------------------------------------------


def all_reduce_scalar(value: float, *, op: str = "mean") -> float:
    """All-reduce a scalar across processes.

    No-op if not distributed (returns *value* unchanged).

    Args:
        value: Scalar to reduce.
        op: ``"mean"`` or ``"sum"``.

    Returns:
        Reduced scalar.
    """
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor)
    if op == "mean":
        tensor /= dist.get_world_size()
    return float(tensor.item())


class _GatherWithGrad(torch.autograd.Function):
    """All-gather that preserves gradient flow for contrastive loss.

    During forward, gathers tensors from all ranks.  During backward,
    copies the local rank's gradient slice back to the input.
    """

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ctx.rank = dist.get_rank()
        ctx.batch_size = tensor.shape[0]
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
        start = ctx.rank * ctx.batch_size
        end = start + ctx.batch_size
        return grad_output[start:end]


def all_gather_tensors(
    tensor: torch.Tensor,
    *,
    with_grad: bool = True,
) -> torch.Tensor:
    """Gather tensors from all ranks and concatenate on dim 0.

    No-op if not distributed (returns *tensor* unchanged).

    Args:
        tensor: ``(B, D)`` tensor to gather.
        with_grad: If ``True``, use a custom autograd function so
            gradients from negative pairs on remote GPUs flow back
            to the local model.  Critical for contrastive loss quality.

    Returns:
        ``(B * world_size, D)`` concatenated tensor.
    """
    if not dist.is_initialized():
        return tensor
    if with_grad:
        return _GatherWithGrad.apply(tensor)
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def gather_string_lists(
    strings: list[str],
    world_size: int,
) -> list[str]:
    """Gather lists of strings from all ranks.

    No-op if not distributed.

    Args:
        strings: Local list of strings.
        world_size: Total process count.

    Returns:
        Concatenated list from all ranks.
    """
    if not dist.is_initialized():
        return strings
    gathered: list[list[str]] = [[] for _ in range(world_size)]
    dist.all_gather_object(gathered, strings)
    result: list[str] = []
    for g in gathered:
        result.extend(g)
    return result


# ------------------------------------------------------------------
# LogitScaleModule — wraps the learnable temperature for DDP
# ------------------------------------------------------------------


class LogitScaleModule(nn.Module):
    """Wraps the learnable logit-scale parameter as an ``nn.Module``.

    DDP can only wrap ``nn.Module`` instances, not raw ``nn.Parameter``.
    This thin wrapper lets DDP synchronize the temperature gradient.

    Args:
        init_value: Initial value for the log-scale parameter.
            Default ``2.6593`` is ``ln(1/0.07)`` (CLIP default).
        device: Target device.
    """

    def __init__(
        self,
        *,
        init_value: float = 2.6593,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, device=device))

    def forward(self) -> torch.Tensor:
        """Return the log-scale parameter value."""
        return self.scale
