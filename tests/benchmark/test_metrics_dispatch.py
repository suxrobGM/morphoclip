"""Pin what benchmark.metrics asks copairs for.

metrics.py is 307 lines with no test. Its actual semantics are the four column
tuples each evaluate_* function passes down (pos_sameby / pos_diffby /
neg_sameby / neg_diffby): those decide what counts as a replicate, a match and a
cross-modality pair, and getting one wrong changes every published number without
raising anything.

Everything here stubs the copairs backend, so no optional extra is needed and
the tests stay fast.
"""

import pandas as pd
import pytest

from benchmark import copairs_backend, metrics, stable_map

PROFILES = pd.DataFrame(
    {
        "Metadata_broad_sample": ["BRD-A", "BRD-A", "BRD-B", "BRD-B"],
        "Metadata_Plate": ["P1", "P2", "P1", "P2"],
        "Metadata_negcon": [0, 0, 0, 0],
        "Metadata_matching_target": ["T1", "T1", "T2", "T2"],
        "Metadata_modality": ["compound", "compound", "crispr", "crispr"],
        "feature_0000": [0.1, 0.2, 0.3, 0.4],
        "feature_0001": [0.5, 0.6, 0.7, 0.8],
    }
)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Intercept the legacy pipeline and record how it was called."""
    seen: dict = {}

    def fake_legacy(
        meta,
        features,
        pos_sameby,
        pos_diffby,
        neg_sameby,
        neg_diffby,
        null_size,
        batch_size,
        use_abs,
        multilabel_col,
    ):
        seen.update(
            meta=meta,
            features=features,
            pos_sameby=pos_sameby,
            pos_diffby=pos_diffby,
            neg_sameby=neg_sameby,
            neg_diffby=neg_diffby,
            null_size=null_size,
            batch_size=batch_size,
            use_abs=use_abs,
            multilabel_col=multilabel_col,
        )
        return pd.DataFrame({"average_precision": [0.5] * len(meta)})

    monkeypatch.setattr(stable_map, "_run_legacy_pipeline", fake_legacy)
    return seen


def test_replicability_pairs_replicates_within_a_plate(captured: dict) -> None:
    """Same perturbation, same plate, differing negcon flag."""
    metrics.evaluate_replicability(PROFILES)

    assert captured["pos_sameby"] == ["Metadata_broad_sample"]
    assert captured["pos_diffby"] == []
    assert captured["neg_sameby"] == ["Metadata_Plate"]
    assert captured["neg_diffby"] == ["Metadata_negcon"]
    assert captured["use_abs"] is False
    assert captured["multilabel_col"] is None


def test_replicability_drops_negative_controls(captured: dict) -> None:
    profiles = PROFILES.assign(Metadata_negcon=[0, 1, 0, 1])
    result = metrics.evaluate_replicability(profiles)
    assert captured, "backend was never reached"
    assert len(result) == 2


def test_replicability_defaults_negcon_to_zero_when_absent(captured: dict) -> None:
    """No negcon column anywhere means keep every row rather than crash."""
    profiles = PROFILES.drop(columns=["Metadata_negcon"])
    result = metrics.evaluate_replicability(profiles)
    assert captured["neg_diffby"] == ["Metadata_negcon"]
    assert len(result) == len(profiles)
    assert (result["Metadata_negcon"] == 0).all()


def test_matching_uses_absolute_similarity_and_multilabel(captured: dict) -> None:
    """Target matching is anti-correlation aware, unlike replicability."""
    metrics.evaluate_matching(PROFILES)

    assert captured["pos_sameby"] == ["Metadata_matching_target"]
    assert captured["neg_diffby"] == ["Metadata_matching_target"]
    assert captured["use_abs"] is True
    assert captured["multilabel_col"] == "Metadata_matching_target"


def test_matching_without_multilabel_passes_no_multilabel_column(captured: dict) -> None:
    metrics.evaluate_matching(PROFILES, multilabel=False)
    assert captured["multilabel_col"] is None


def test_cross_modality_requires_different_modalities(captured: dict) -> None:
    """The defining difference: positives must share a target but differ in modality."""
    metrics.evaluate_cross_modality_matching(PROFILES)

    assert captured["pos_sameby"] == ["Metadata_matching_target"]
    assert captured["pos_diffby"] == ["Metadata_modality"]
    assert captured["neg_diffby"] == ["Metadata_matching_target", "Metadata_modality"]
    assert captured["use_abs"] is True


def test_only_feature_columns_reach_the_backend(captured: dict) -> None:
    metrics.evaluate_replicability(PROFILES)
    assert captured["features"].shape == (4, 2)
    assert list(captured["meta"].columns) == [
        c for c in PROFILES.columns if c.startswith("Metadata_")
    ]


def test_fraction_retrieved_counts_significant_groups() -> None:
    map_df = pd.DataFrame({"above_q_threshold": [True, False, True, True]})
    assert metrics.compute_fraction_retrieved(map_df) == 0.75


def test_fraction_retrieved_divides_by_zero_on_an_empty_frame() -> None:
    """Recorded as-is: an empty mAP frame raises rather than returning 0."""
    empty = pd.DataFrame({"above_q_threshold": pd.Series(dtype=bool)})
    with pytest.raises(ZeroDivisionError):
        metrics.compute_fraction_retrieved(empty)


@pytest.mark.parametrize(
    "legacy_column", ["above_q_threshold", "above_corrected_p_threshold", "below_corrected_p"]
)
def test_compute_map_normalizes_the_significance_column(
    monkeypatch: pytest.MonkeyPatch, legacy_column: str
) -> None:
    """Three copairs versions spell this column three ways; all become above_q_threshold."""
    aggregated = pd.DataFrame({"Metadata_broad_sample": ["BRD-A"], "average_precision": [0.5]})
    aggregated[legacy_column] = [True]

    monkeypatch.setattr(
        copairs_backend,
        "_get_legacy_modules",
        lambda: {"aggregate": lambda *_args, **_kwargs: aggregated},
    )

    map_df = metrics.compute_map(
        pd.DataFrame({"average_precision": [0.5]}),
        ["Metadata_broad_sample"],
    )
    assert "above_q_threshold" in map_df.columns
    assert "mean_average_precision" in map_df.columns
