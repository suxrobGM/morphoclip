"""Tests for benchmark stable-helper utilities."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed (benchmark extra)")

from benchmark.data import ProfileLoader  # noqa: E402
from benchmark.stable_helpers import (  # noqa: E402
    apply_batch_correction,
    fit_batch_correction,
    run_with_unpaired_guard,
)


def _write_plate(tmp_path, batch: str, plate: str, df: pd.DataFrame) -> None:
    plate_dir = tmp_path / batch / plate
    plate_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        plate_dir / f"{plate}_normalized_feature_select_negcon_batch.csv.gz",
        index=False,
        compression="gzip",
    )


def test_batch_correction_fits_on_negcons_and_replaces_feature_space(tmp_path) -> None:
    batch = "2020_11_04_CPJUMP1"
    plate1 = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02", "A03"],
            "Metadata_control_type": ["negcon", "negcon", "trt"],
            "Metadata_broad_sample": ["ctl1", "ctl2", "cmpd1"],
            "feature_0000": [1.0, 2.0, 9.0],
            "feature_0001": [2.0, 1.0, 8.0],
            "feature_0002": [0.5, 0.7, 7.0],
        }
    )
    plate2 = pd.DataFrame(
        {
            "Metadata_Well": ["B01", "B02", "B03"],
            "Metadata_control_type": ["negcon", "negcon", "trt"],
            "Metadata_broad_sample": ["ctl3", "ctl4", "cmpd2"],
            "feature_0000": [1.5, 2.5, 10.0],
            "feature_0001": [2.5, 1.5, 9.0],
            "feature_0002": [0.6, 0.8, 8.0],
        }
    )
    _write_plate(tmp_path, batch, "BR001", plate1)
    _write_plate(tmp_path, batch, "BR002", plate2)

    loader = ProfileLoader(tmp_path)
    transform = fit_batch_correction(
        loader=loader,
        batch=batch,
        plates=["BR001", "BR002"],
        n_components=10,
    )

    assert transform.control_count == 4
    assert transform.requested_n_components == 10
    assert transform.effective_n_components == 3

    combined = pd.concat([plate1, plate2], ignore_index=True)
    corrected = apply_batch_correction(combined, transform)

    assert corrected["Metadata_Well"].tolist() == combined["Metadata_Well"].tolist()
    assert "feature_0000" not in corrected.columns
    assert list(corrected.columns[:3]) == [
        "Metadata_Well",
        "Metadata_control_type",
        "Metadata_broad_sample",
    ]
    assert list(corrected.columns[3:]) == [
        "corrected_feature_0000",
        "corrected_feature_0001",
        "corrected_feature_0002",
    ]

    negcon = corrected.loc[
        corrected["Metadata_control_type"] == "negcon",
        list(transform.output_feature_cols),
    ].to_numpy(dtype=np.float32)
    np.testing.assert_allclose(negcon.mean(axis=0), np.zeros(3, dtype=np.float32), atol=1e-6)


def test_batch_correction_requires_negative_controls(tmp_path) -> None:
    batch = "2020_11_04_CPJUMP1"
    plate = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02"],
            "Metadata_control_type": ["trt", "trt"],
            "Metadata_broad_sample": ["cmpd1", "cmpd2"],
            "feature_0000": [1.0, 2.0],
            "feature_0001": [3.0, 4.0],
        }
    )
    _write_plate(tmp_path, batch, "BR001", plate)

    loader = ProfileLoader(tmp_path)
    with pytest.raises(ValueError, match="no 'negcon' wells"):
        fit_batch_correction(loader=loader, batch=batch, plates=["BR001"])


def test_apply_batch_correction_validates_expected_feature_columns(tmp_path) -> None:
    batch = "2020_11_04_CPJUMP1"
    plate = pd.DataFrame(
        {
            "Metadata_Well": ["A01", "A02", "A03"],
            "Metadata_control_type": ["negcon", "negcon", "trt"],
            "Metadata_broad_sample": ["ctl1", "ctl2", "cmpd1"],
            "feature_0000": [1.0, 2.0, 3.0],
            "feature_0001": [4.0, 5.0, 6.0],
        }
    )
    _write_plate(tmp_path, batch, "BR001", plate)

    loader = ProfileLoader(tmp_path)
    transform = fit_batch_correction(loader=loader, batch=batch, plates=["BR001"], n_components=2)

    broken = plate.drop(columns=["feature_0001"])
    with pytest.raises(ValueError, match="missing batch-correction feature columns"):
        apply_batch_correction(broken, transform)


def test_run_with_unpaired_guard_suppresses_empty_dict_pairs() -> None:
    def _raise() -> pd.DataFrame:
        raise ValueError("dict_pairs empty")

    result = run_with_unpaired_guard(_raise)

    assert result.empty
