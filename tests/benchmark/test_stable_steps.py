"""End-to-end tests for the two benchmark evaluation steps.

These are the modules the 548-test suite did not reach: a refactor left five call
sites passing a removed ``copairs_mode`` keyword, and ``morphoclip benchmark``
died with a ``TypeError`` that no test saw. Everything here runs the real
copairs pipeline over synthetic profiles, so a broken call site fails in CI.
"""

from pathlib import Path

import pandas as pd
import pytest

from benchmark import stable_map
from benchmark.data import ProfileLoader
from benchmark.stable_compound import CompoundContext, evaluate_compound
from benchmark.stable_config import BenchmarkParams
from benchmark.stable_genetic import evaluate_genetic
from benchmark.stable_results import StableResults
from tests.support.profiles import (
    BATCH,
    Perturbation,
    make_experiments_frame,
    write_profile_tree,
)

GENES = ["GENE0", "GENE1", "GENE2", "GENE3"]
COMPOUND_PLATES = ["BR00116991", "BR00116992"]
CRISPR_PLATES = ["BR00117000", "BR00117001"]

# Two compounds per gene, so a target has more than one member to retrieve.
COMPOUNDS = [Perturbation(f"BRD-K{i:05d}", gene) for i, gene in enumerate(GENES * 2)]
# Two guides per gene, so no gene is dropped by the sister-guide filter.
GUIDES = [Perturbation(f"{gene}-guide{i // len(GENES)}", gene) for i, gene in enumerate(GENES * 2)]


@pytest.fixture(autouse=True)
def _serial_copairs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Take the serial branch copairs already falls back to on sandboxed runners.

    copairs spawns a ``multiprocessing.Pool`` per null-distribution call. Under
    pytest each spawn re-imports the world, which costs ~4s per test for
    microseconds of arithmetic. The fallback computes the same numbers.
    """
    monkeypatch.setattr(
        stable_map,
        "_run_with_serial_fallback",
        lambda primary, fallback: fallback(),
    )


@pytest.fixture(scope="module")
def profiles_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A profile tree with two compound plates and two CRISPR plates."""
    root = tmp_path_factory.mktemp("profiles")
    write_profile_tree(
        root,
        dict.fromkeys(COMPOUND_PLATES, COMPOUNDS) | dict.fromkeys(CRISPR_PLATES, GUIDES),
    )
    return root


@pytest.fixture
def params(profiles_root: Path, tmp_path: Path) -> BenchmarkParams:
    return BenchmarkParams(
        project_root=tmp_path,
        profiles_path=profiles_root,
        output_path=tmp_path / "out",
        figures_dir=tmp_path / "out" / "figures",
        tables_dir=tmp_path / "out" / "tables",
        batch=BATCH,
        test_mode=False,
        cell_filter=None,
        batch_correction=False,
        pca_kernel="linear",
        pca_n_components=8,
        timelines=["short"],
        split_manifest_path=None,
        subset=None,
        manifest=None,
    )


@pytest.fixture
def loader(profiles_root: Path) -> ProfileLoader:
    return ProfileLoader(profiles_root)


@pytest.fixture
def experiments() -> pd.DataFrame:
    return make_experiments_frame(
        [
            {"Assay_Plate_Barcode": p, "Perturbation": "compound", "Cell_type": "U2OS", "Time": 24}
            for p in COMPOUND_PLATES
        ]
        + [
            {"Assay_Plate_Barcode": p, "Perturbation": "crispr", "Cell_type": "U2OS", "Time": 96}
            for p in CRISPR_PLATES
        ]
    )


@pytest.fixture
def target_metadata() -> pd.DataFrame:
    """The compound-to-target annotation table, pipe-separated as on disk."""
    return pd.DataFrame(
        {
            "Metadata_broad_sample": [c.broad_sample for c in COMPOUNDS],
            "Metadata_target_list": [c.gene for c in COMPOUNDS],
        }
    )


def _run_compound(
    results: StableResults,
    *,
    loader: ProfileLoader,
    params: BenchmarkParams,
    experiments: pd.DataFrame,
    target_metadata: pd.DataFrame,
) -> CompoundContext | None:
    return evaluate_compound(
        results,
        loader=loader,
        params=params,
        correction_transform=None,
        target1_metadata=target_metadata,
        cell="U2OS",
        modality_1_experiments_df=experiments.query("Perturbation=='compound'"),
        modality_1_timepoint=24,
        count=1,
        batch_size=1000,
        null_size=200,
    )


class TestEvaluateCompound:
    def test_appends_replicability_and_matching_rows(
        self,
        loader: ProfileLoader,
        params: BenchmarkParams,
        experiments: pd.DataFrame,
        target_metadata: pd.DataFrame,
    ) -> None:
        results = StableResults()
        context = _run_compound(
            results,
            loader=loader,
            params=params,
            experiments=experiments,
            target_metadata=target_metadata,
        )

        assert context is not None
        assert context.perturbation == "compound"
        assert context.time_label == "short"
        assert len(results.replicability_map) == len(COMPOUNDS)
        assert results.replicability_fr.loc[0, "Description"] == "compound_U2OS_short"
        # Replicates are near-copies of one signature, so every compound retrieves.
        assert results.replicability_fr.loc[0, "fr"] == 1.0
        assert set(results.matching_map["Metadata_matching_target"]) == set(GENES)

    def test_returns_none_when_the_timepoint_has_no_plates(
        self,
        loader: ProfileLoader,
        params: BenchmarkParams,
        experiments: pd.DataFrame,
        target_metadata: pd.DataFrame,
    ) -> None:
        results = StableResults()
        context = evaluate_compound(
            results,
            loader=loader,
            params=params,
            correction_transform=None,
            target1_metadata=target_metadata,
            cell="U2OS",
            modality_1_experiments_df=experiments.query("Perturbation=='compound'"),
            modality_1_timepoint=999,
            count=1,
            batch_size=1000,
            null_size=200,
        )

        assert context is None
        assert results.replicability_map.empty

    def test_consensus_holds_one_row_per_replicable_compound(
        self,
        loader: ProfileLoader,
        params: BenchmarkParams,
        experiments: pd.DataFrame,
        target_metadata: pd.DataFrame,
    ) -> None:
        results = StableResults()
        context = _run_compound(
            results,
            loader=loader,
            params=params,
            experiments=experiments,
            target_metadata=target_metadata,
        )

        assert context is not None
        consensus = context.modality_1_consensus_df
        assert len(consensus) == len(COMPOUNDS)
        assert consensus["Metadata_broad_sample"].is_unique
        # The pipe-separated annotation becomes a list for multilabel matching.
        assert all(isinstance(t, list) for t in consensus["Metadata_matching_target"])
        assert "Metadata_target_list" not in consensus.columns
        # Negative controls must not survive into the consensus.
        assert "DMSO" not in set(consensus["Metadata_broad_sample"])


class TestEvaluateGenetic:
    def test_appends_crispr_and_cross_modality_rows(
        self,
        loader: ProfileLoader,
        params: BenchmarkParams,
        experiments: pd.DataFrame,
        target_metadata: pd.DataFrame,
    ) -> None:
        results = StableResults()
        context = _run_compound(
            results,
            loader=loader,
            params=params,
            experiments=experiments,
            target_metadata=target_metadata,
        )
        assert context is not None

        evaluate_genetic(
            results,
            loader=loader,
            params=params,
            correction_transform=None,
            cell="U2OS",
            cell_df=experiments,
            compound=context,
            count=2,
            batch_size=1000,
            null_size=200,
        )

        descriptions = set(results.replicability_fr["Description"])
        assert descriptions == {"compound_U2OS_short", "crispr_U2OS_short"}
        assert "crispr_U2OS_short" in set(results.matching_fr["Description"])
        assert list(results.gene_compound_matching_fr["Description"]) == [
            "compound_U2OS_short-crispr_U2OS_short"
        ]
        assert list(results.gene_compound_matching_fr["Modality1"]) == ["compound"]
        assert list(results.gene_compound_matching_fr["Modality2"]) == ["crispr"]

    def test_skips_a_modality_it_has_already_computed(
        self,
        loader: ProfileLoader,
        params: BenchmarkParams,
        experiments: pd.DataFrame,
        target_metadata: pd.DataFrame,
    ) -> None:
        results = StableResults()
        context = _run_compound(
            results,
            loader=loader,
            params=params,
            experiments=experiments,
            target_metadata=target_metadata,
        )
        assert context is not None

        def run() -> None:
            evaluate_genetic(
                results,
                loader=loader,
                params=params,
                correction_transform=None,
                cell="U2OS",
                cell_df=experiments,
                compound=context,
                count=2,
                batch_size=1000,
                null_size=200,
            )

        run()
        replicability_rows = len(results.replicability_map)
        run()

        # The replicability step is guarded by _already_computed; cross-modality is not.
        assert len(results.replicability_map) == replicability_rows
        assert len(results.gene_compound_matching_fr) == 2
