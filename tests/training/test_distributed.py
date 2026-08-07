"""Tests for the distributed helpers in single-process mode."""

import torch

from morphoclip.training.distributed import (
    all_gather_tensors,
    all_reduce_scalar,
    broadcast_flag,
    gather_string_lists,
    setup_distributed,
)


class TestSingleProcessFallbacks:
    """Without a process group every collective must pass its input through."""

    def test_broadcast_flag_returns_local_value(self) -> None:
        assert broadcast_flag(True) is True
        assert broadcast_flag(False) is False

    def test_all_reduce_scalar_is_identity(self) -> None:
        assert all_reduce_scalar(1.25) == 1.25

    def test_all_gather_tensors_is_identity(self) -> None:
        tensor = torch.randn(3, 4)
        assert all_gather_tensors(tensor) is tensor

    def test_gather_string_lists_is_identity(self) -> None:
        strings = ["a", "b"]
        assert gather_string_lists(strings, 1) is strings


def test_setup_distributed_without_torchrun(monkeypatch) -> None:
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    state = setup_distributed()

    assert state.rank == 0
    assert state.world_size == 1
    assert state.is_main
