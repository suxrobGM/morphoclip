"""Result accumulation, persistence, and reporting for the stable benchmark.

Holds the six mAP/fraction-retrieved accumulator frames and collapses the
repeated build-and-concat pattern into :meth:`StableResults.append`. The heavy
copairs (:func:`compute_map_and_fr`) and matplotlib (plot) dependencies are
imported lazily inside the methods that need them, so this module — and the pure
accumulation/persistence logic — can be imported without the ``benchmark`` extra.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from benchmark.profile_ops import concat_profiles


def _already_computed(df: pd.DataFrame, description: str) -> bool:
    """Whether *df* already contains rows for *description*."""
    return not df.empty and bool(df.Description.str.contains(description).any())


def _write_pivot(df: pd.DataFrame, *, index: list[str], path: Path) -> None:
    """Write a Cell-columned fraction-retrieved pivot table, skipping empty frames."""
    if df.empty:
        return
    pivot = df.pivot_table(values="fr", index=index, columns="Cell", aggfunc="first")
    pivot.to_csv(path)


@dataclass
class StableResults:
    """Six accumulator frames for replicability / matching / cross-modality mAP+FR."""

    replicability_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    replicability_fr: pd.DataFrame = field(default_factory=pd.DataFrame)
    matching_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    matching_fr: pd.DataFrame = field(default_factory=pd.DataFrame)
    gene_compound_matching_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    gene_compound_matching_fr: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(
        self,
        map_attr: str,
        fr_attr: str,
        *,
        result: pd.DataFrame,
        group_cols: list[str],
        null_size: int,
        metadata: dict[str, object],
    ) -> None:
        """Compute mAP/FR for *result* and append a row to each accumulator.

        *metadata* is applied verbatim as columns; its insertion order determines
        the column order (fraction-retrieved rows append ``fr`` last).
        """
        from benchmark.stable_helpers import compute_map_and_fr

        map_df, fr = compute_map_and_fr(result, group_cols, null_size)
        self._append_rows(map_attr, fr_attr, map_df=map_df, fr=fr, metadata=metadata)

    def _append_rows(
        self,
        map_attr: str,
        fr_attr: str,
        *,
        map_df: pd.DataFrame,
        fr: float,
        metadata: dict[str, object],
    ) -> None:
        """Attach *metadata* to the mAP/FR rows and concat (pure; no copairs)."""
        fr_row = {key: [value] for key, value in metadata.items()}
        fr_row["fr"] = [fr]
        setattr(self, fr_attr, concat_profiles(getattr(self, fr_attr), pd.DataFrame(fr_row)))

        for key, value in metadata.items():
            map_df[key] = value
        setattr(self, map_attr, concat_profiles(getattr(self, map_attr), map_df))

    def save(self, output_path: Path, tables_dir: Path) -> None:
        """Write the six accumulator CSVs plus the three fraction-retrieved pivots."""
        print("\nSaving results...")

        self.replicability_map.to_csv(
            output_path / "cellprofiler_replicability_map.csv", index=False
        )
        self.replicability_fr.to_csv(output_path / "cellprofiler_replicability_fr.csv", index=False)
        self.matching_map.to_csv(output_path / "cellprofiler_matching_map.csv", index=False)
        self.matching_fr.to_csv(output_path / "cellprofiler_matching_fr.csv", index=False)
        self.gene_compound_matching_map.to_csv(
            output_path / "cellprofiler_gene_compound_matching_map.csv", index=False
        )
        self.gene_compound_matching_fr.to_csv(
            output_path / "cellprofiler_gene_compound_matching_fr.csv", index=False
        )

        _write_pivot(
            self.replicability_fr,
            index=["Modality", "time"],
            path=tables_dir / "replicability_summary.csv",
        )
        _write_pivot(
            self.matching_fr,
            index=["Modality", "time"],
            path=tables_dir / "matching_summary.csv",
        )
        _write_pivot(
            self.gene_compound_matching_fr,
            index=["Modality1", "Modality2"],
            path=tables_dir / "gene_compound_matching_summary.csv",
        )

    def generate_figures(self, figures_dir: Path) -> None:
        """Render the benchmark figures (requires matplotlib/seaborn)."""
        from benchmark.stable_helpers import (
            plot_cross_modality_barplot,
            plot_matching_barplot,
            plot_matching_fr_faceted,
            plot_matching_map_boxplot,
            plot_replicability_barplot,
            plot_replicability_fr_faceted,
            plot_replicability_map_boxplot,
        )

        print("\nGenerating figures...")

        if not self.replicability_fr.empty:
            plot_replicability_barplot(
                self.replicability_fr, figures_dir / "replicability_fr_barplot.png"
            )
            plot_replicability_fr_faceted(
                self.replicability_fr, figures_dir / "replicability_fr_faceted.png"
            )

        if not self.replicability_map.empty:
            plot_replicability_map_boxplot(
                self.replicability_map, figures_dir / "replicability_map_boxplot.png"
            )

        if not self.matching_fr.empty:
            plot_matching_barplot(self.matching_fr, figures_dir / "matching_fr_barplot.png")
            plot_matching_fr_faceted(self.matching_fr, figures_dir / "matching_fr_faceted.png")

        if not self.matching_map.empty:
            plot_matching_map_boxplot(self.matching_map, figures_dir / "matching_map_boxplot.png")

        if not self.gene_compound_matching_fr.empty:
            plot_cross_modality_barplot(
                self.gene_compound_matching_fr,
                figures_dir / "gene_compound_matching_fr_barplot.png",
            )

    def print_summary(self, output_path: Path) -> None:
        """Print the fraction-retrieved summary tables and output location."""
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        if not self.replicability_fr.empty:
            print("\nReplicability (Fraction Retrieved):")
            print(self.replicability_fr[["Description", "fr"]].to_string(index=False))

        if not self.matching_fr.empty:
            print("\nTarget Matching (Fraction Retrieved):")
            print(self.matching_fr[["Description", "fr"]].to_string(index=False))

        if not self.gene_compound_matching_fr.empty:
            print("\nGene-Compound Matching (Fraction Retrieved):")
            print(self.gene_compound_matching_fr[["Description", "fr"]].to_string(index=False))

        print("\n" + "=" * 60)
        print(f"Results saved to: {output_path}")
        print("=" * 60)
