"""Unit tests for benchmark.stable_results (pure; no copairs/sklearn needed).

Exercises the accumulation schema (fr-vs-map columns + order), the join='inner'
concat behavior, the already-computed guard, and CSV/pivot persistence — the
logic introduced by collapsing the repeated build-and-concat blocks.
"""

import pandas as pd

from benchmark.stable_results import StableResults, _already_computed, _write_pivot


class TestAlreadyComputed:
    def test_empty_is_false(self):
        assert _already_computed(pd.DataFrame(), "crispr_A549_long") is False

    def test_present_is_true(self):
        df = pd.DataFrame({"Description": ["crispr_A549_long", "orf_A549_short"]})
        assert _already_computed(df, "crispr_A549_long") is True

    def test_absent_is_false(self):
        df = pd.DataFrame({"Description": ["orf_A549_short"]})
        assert _already_computed(df, "crispr_A549_long") is False


def _map_df(**extra) -> pd.DataFrame:
    """A minimal mAP frame like compute_map would return."""
    base = {
        "Metadata_broad_sample": ["c1", "c2"],
        "mean_average_precision": [0.4, 0.6],
        "above_q_threshold": [True, False],
    }
    base.update(extra)
    return pd.DataFrame(base)


class TestAppendRows:
    def test_fr_row_schema_and_order(self):
        results = StableResults()
        results._append_rows(
            "replicability_map",
            "replicability_fr",
            map_df=_map_df(),
            fr=0.75,
            metadata={
                "Description": "compound_A549_long",
                "Modality": "compound",
                "Cell": "A549",
                "time": "long",
                "timepoint": 48,
            },
        )
        assert list(results.replicability_fr.columns) == [
            "Description",
            "Modality",
            "Cell",
            "time",
            "timepoint",
            "fr",
        ]
        assert results.replicability_fr.iloc[0]["fr"] == 0.75

    def test_map_metadata_columns_appended(self):
        results = StableResults()
        results._append_rows(
            "replicability_map",
            "replicability_fr",
            map_df=_map_df(),
            fr=0.5,
            metadata={
                "Description": "compound_A549_long",
                "Modality": "compound",
                "Cell": "A549",
                "time": "long",
                "timepoint": 48,
            },
        )
        cols = results.replicability_map.columns
        for key in ("Description", "Modality", "Cell", "time", "timepoint"):
            assert key in cols
        assert "mean_average_precision" in cols
        assert len(results.replicability_map) == 2

    def test_concat_uses_inner_join(self):
        """A second append with an extra map column drops it (join='inner')."""
        results = StableResults()
        meta = {
            "Description": "a",
            "Modality": "compound",
            "Cell": "A549",
            "time": "long",
            "timepoint": 48,
        }
        results._append_rows(
            "replicability_map", "replicability_fr", map_df=_map_df(), fr=0.5, metadata=meta
        )
        results._append_rows(
            "replicability_map",
            "replicability_fr",
            map_df=_map_df(extra_col=[1, 2]),
            fr=0.6,
            metadata={**meta, "Description": "b"},
        )
        # inner join keeps only columns common to both appends.
        assert "extra_col" not in results.replicability_map.columns
        assert len(results.replicability_map) == 4

    def test_cross_modality_schema(self):
        results = StableResults()
        results._append_rows(
            "gene_compound_matching_map",
            "gene_compound_matching_fr",
            map_df=_map_df(),
            fr=0.3,
            metadata={
                "Description": "x-y",
                "Modality1": "compound",
                "Modality2": "crispr",
                "Cell": "A549",
                "time1": "long",
                "time2": "short",
                "timepoint1": 48,
                "timepoint2": 96,
            },
        )
        assert list(results.gene_compound_matching_fr.columns) == [
            "Description",
            "Modality1",
            "Modality2",
            "Cell",
            "time1",
            "time2",
            "timepoint1",
            "timepoint2",
            "fr",
        ]


class TestWritePivot:
    def test_empty_writes_nothing(self, tmp_path):
        out = tmp_path / "summary.csv"
        _write_pivot(pd.DataFrame(), index=["Modality", "time"], path=out)
        assert not out.exists()

    def test_non_empty_writes(self, tmp_path):
        out = tmp_path / "summary.csv"
        df = pd.DataFrame(
            {"Modality": ["compound"], "time": ["long"], "Cell": ["A549"], "fr": [0.5]}
        )
        _write_pivot(df, index=["Modality", "time"], path=out)
        assert out.exists()


class TestSave:
    def test_writes_six_csvs(self, tmp_path):
        results = StableResults()
        meta = {
            "Description": "a",
            "Modality": "compound",
            "Cell": "A549",
            "time": "long",
            "timepoint": 48,
        }
        results._append_rows(
            "replicability_map", "replicability_fr", map_df=_map_df(), fr=0.5, metadata=meta
        )
        output_path = tmp_path / "out"
        tables_dir = tmp_path / "tables"
        output_path.mkdir()
        tables_dir.mkdir()

        results.save(output_path, tables_dir)

        for name in (
            "cellprofiler_replicability_map.csv",
            "cellprofiler_replicability_fr.csv",
            "cellprofiler_matching_map.csv",
            "cellprofiler_matching_fr.csv",
            "cellprofiler_gene_compound_matching_map.csv",
            "cellprofiler_gene_compound_matching_fr.csv",
        ):
            assert (output_path / name).exists()
        assert (tables_dir / "replicability_summary.csv").exists()
