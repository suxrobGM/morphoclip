"""Tests for benchmark metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark import copairs_backend, stable_map


def test_compute_legacy_similarities_falls_back_to_serial(monkeypatch) -> None:
    class Backend:
        @staticmethod
        def pairwise_cosine(x_sample: np.ndarray, y_sample: np.ndarray) -> np.ndarray:
            x_norm = x_sample / np.linalg.norm(x_sample, axis=1, keepdims=True)
            y_norm = y_sample / np.linalg.norm(y_sample, axis=1, keepdims=True)
            return np.sum(x_norm * y_norm, axis=1)

    monkeypatch.setattr(
        copairs_backend,
        "_get_legacy_modules",
        lambda: {
            "backend": Backend,
            "cosine_indexed": lambda *_args, **_kwargs: (_ for _ in ()).throw(
                PermissionError("[Errno 13] Permission denied")
            ),
        },
    )

    pairs = pd.DataFrame({"ix1": [0, 0, 1], "ix2": [1, 2, 2]})
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = stable_map._compute_legacy_similarities(
        pairs=pairs,
        features=features,
        batch_size=2,
        use_abs=False,
    )

    assert result["dist"].tolist() == [1.0, 0.0, 0.0]


def test_compute_legacy_null_dists_falls_back_to_serial(monkeypatch) -> None:
    class Backend:
        @staticmethod
        def compute_null_dists(_rel_k_list: pd.Series, _null_size: int) -> np.ndarray:
            raise PermissionError("[Errno 13] Permission denied")

        @staticmethod
        def random_ap(num_perm: int, num_pos: int, num_neg: int, seed=None) -> np.ndarray:
            return np.full(num_perm, num_pos + num_neg, dtype=float)

    monkeypatch.setattr(
        copairs_backend,
        "_get_legacy_modules",
        lambda: {
            "backend": Backend,
        },
    )

    rel_k_list = pd.Series(
        [
            np.array([[1, 0, 1], [0, 1, 0]]),
            np.array([[1, 1, 0]]),
        ]
    )

    null_dists = stable_map._compute_legacy_null_dists(rel_k_list, null_size=4)

    assert null_dists.shape == (2, 4)
    # rel_k_list[0]: 2x3 array, sum=3 positives, size=6, neg=3 -> random_ap(4, 3, 3) -> [6]*4
    assert null_dists[0].tolist() == [6.0, 6.0, 6.0, 6.0]
    # rel_k_list[1]: 1x3 array, sum=2 positives, size=3, neg=1 -> random_ap(4, 2, 1) -> [3]*4
    assert null_dists[1].tolist() == [3.0, 3.0, 3.0, 3.0]


def test_compute_legacy_similarities_handles_empty_pairs() -> None:
    try:
        copairs_backend._get_legacy_modules()
    except RuntimeError:
        pytest.skip("Legacy copairs not installed")

    features = np.array([[1.0, 0.0]])
    pairs = pd.DataFrame(columns=["ix1", "ix2"])

    result = stable_map._compute_legacy_similarities(
        pairs=pairs,
        features=features,
        batch_size=2,
        use_abs=False,
    )

    assert list(result.columns) == ["ix1", "ix2", "dist"]
    assert result.empty
