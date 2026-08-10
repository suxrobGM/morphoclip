"""Tests for the distributed helpers in single-process mode."""

import pytest
import torch

from morphoclip.training.distributed import (
    all_gather_tensors,
    all_reduce_scalar,
    broadcast_flag,
    gather_string_lists,
    setup_distributed,
)


def test_collectives_pass_their_input_through_without_a_process_group() -> None:
    tensor = torch.randn(3, 4)
    strings = ["a", "b"]

    assert broadcast_flag(True) is True
    assert broadcast_flag(False) is False
    assert all_reduce_scalar(1.25) == 1.25
    assert all_gather_tensors(tensor) is tensor
    assert gather_string_lists(strings, 1) is strings


def test_setup_distributed_without_torchrun(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)

    state = setup_distributed()

    assert state.rank == 0
    assert state.world_size == 1
    assert state.is_main
