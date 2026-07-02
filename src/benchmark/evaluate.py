"""Main evaluation pipeline for benchmarking perturbation embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from benchmark.data import (
    ProfileLoader,
    add_negcon_indicator,
    compute_consensus,
    filter_experiment_metadata_to_split_subset,
    filter_profiles_to_split_subset,
    filter_replicable,
    get_timepoint_label,
    load_split_manifest,
    remove_empty_wells,
    remove_negcon_wells,
)
from benchmark.metrics import (
    CopairsMode,
    compute_fraction_retrieved,
    compute_map,
    evaluate_cross_modality_matching,
    evaluate_matching,
    evaluate_replicability,
)


@dataclass
class EvaluationConfig:
    """Configuration for benchmark evaluation."""

    batch: str = "2020_11_04_CPJUMP1"
    profiles_dir: str = "../data/profiles"
    output_dir: str = "output"
    figures_dir: str = "figures"

    experiment_metadata_path: str = "input/experiment-metadata.tsv"
    target_annotations_path: str = (
        "input/JUMP-Target-1_compound_metadata_additional_annotations.tsv"
    )

    density_filter: int = 100
    antibiotics_filter: str = "absent"
    q_threshold: float = 0.05

    null_size: int = 100000
    batch_size: int = 100000
    copairs_mode: CopairsMode = "experimental"

    sample_col: str = "Metadata_broad_sample"
    target_col: str = "Metadata_matching_target"
    split_manifest_path: str | None = None
    split_subset: str | None = None


@dataclass
class EvaluationResults:
    """Container for all evaluation results."""

    replicability_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    replicability_fr: pd.DataFrame = field(default_factory=pd.DataFrame)
    matching_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    matching_fr: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_modality_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_modality_fr: pd.DataFrame = field(default_factory=pd.DataFrame)

    def save(self, output_dir: str | Path, prefix: str = ""):
        """Save all results to CSV files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{prefix}_" if prefix else ""

        self.replicability_map.to_csv(output_dir / f"{prefix}replicability_map.csv", index=False)
        self.replicability_fr.to_csv(output_dir / f"{prefix}replicability_fr.csv", index=False)
        self.matching_map.to_csv(output_dir / f"{prefix}matching_map.csv", index=False)
        self.matching_fr.to_csv(output_dir / f"{prefix}matching_fr.csv", index=False)
        self.cross_modality_map.to_csv(output_dir / f"{prefix}cross_modality_map.csv", index=False)
        self.cross_modality_fr.to_csv(output_dir / f"{prefix}cross_modality_fr.csv", index=False)


class BenchmarkEvaluator:
    """Evaluator for Cell Painting benchmark tasks."""

    def __init__(self, config: EvaluationConfig | None = None):
        """Initialize evaluator with configuration.

        Args:
            config: Evaluation configuration. Uses defaults if not provided.
        """
        self.config = config or EvaluationConfig()
        self.loader = ProfileLoader(self.config.profiles_dir)
        self.results = EvaluationResults()

        self._experiment_df: pd.DataFrame | None = None
        self._target_annotations: pd.DataFrame | None = None
        self._split_manifest: pd.DataFrame | None = None

    @property
    def split_manifest(self) -> pd.DataFrame | None:
        """Load the requested split subset manifest, if configured."""
        if self.config.split_manifest_path is None:
            return None
        if self.config.split_subset is None:
            raise ValueError("split_subset must be provided when split_manifest_path is set")
        if self._split_manifest is None:
            self._split_manifest = load_split_manifest(
                self.config.split_manifest_path,
                self.config.split_subset,
            )
        return self._split_manifest

    @property
    def experiment_df(self) -> pd.DataFrame:
        """Load and filter experiment metadata."""
        if self._experiment_df is None:
            df = pd.read_csv(self.config.experiment_metadata_path, sep="\t")
            df = df.query(f"Batch=='{self.config.batch}'")
            df = df.query(f"Density=={self.config.density_filter}")
            df = df.query(f'Antibiotics=="{self.config.antibiotics_filter}"')
            # Exclude Cas9 cell line for compounds
            df = df.drop(df[(df.Perturbation == "compound") & (df.Cell_line == "Cas9")].index)
            df = filter_experiment_metadata_to_split_subset(df, self.split_manifest)
            self._experiment_df = df
        return self._experiment_df

    @property
    def target_annotations(self) -> pd.DataFrame:
        """Load compound target annotations."""
        if self._target_annotations is None:
            self._target_annotations = pd.read_csv(
                self.config.target_annotations_path,
                sep="\t",
                usecols=["broad_sample", "target_list"],
            ).rename(
                columns={
                    "broad_sample": self.config.sample_col,
                    "target_list": "Metadata_target_list",
                }
            )
        return self._target_annotations

    def get_experiments(
        self,
        cell_type: str,
        perturbation: str,
        timepoint: int | None = None,
    ) -> pd.DataFrame:
        """Get filtered experiment metadata."""
        df = self.experiment_df.query(
            f"Cell_type=='{cell_type}' and Perturbation=='{perturbation}'"
        )
        if timepoint is not None:
            df = df.query(f"Time=={timepoint}")
        return df

    def load_modality_profiles(
        self,
        cell_type: str,
        perturbation: str,
        timepoint: int,
    ) -> pd.DataFrame:
        """Load all profiles for a modality/timepoint combination."""
        exp_df = self.get_experiments(cell_type, perturbation, timepoint)
        plates = exp_df.Assay_Plate_Barcode.unique()

        profiles = self.loader.load_plates(self.config.batch, plates, modality=perturbation)
        profiles = filter_profiles_to_split_subset(profiles, self.split_manifest)

        # Fill missing sample IDs with DMSO for negative controls
        profiles[self.config.sample_col] = profiles[self.config.sample_col].fillna("DMSO")

        return profiles

    def evaluate_modality_replicability(
        self,
        profiles: pd.DataFrame,
        modality: str,
        cell_type: str,
        timepoint: int,
    ) -> tuple[pd.DataFrame, float]:
        """Evaluate replicability for a single modality.

        Returns:
            Tuple of (mAP DataFrame, fraction retrieved).
        """
        profiles = remove_empty_wells(profiles)
        profiles = add_negcon_indicator(profiles)

        result = evaluate_replicability(
            profiles,
            null_size=self.config.null_size,
            batch_size=self.config.batch_size,
            copairs_mode=self.config.copairs_mode,
        )

        map_df = compute_map(
            result,
            [self.config.sample_col],
            self.config.q_threshold,
            null_size=self.config.null_size,
            copairs_mode=self.config.copairs_mode,
        )
        fr = compute_fraction_retrieved(map_df)

        # Add metadata
        time_label = get_timepoint_label(modality, timepoint)
        description = f"{modality}_{cell_type}_{time_label}"

        map_df["Description"] = description
        map_df["Modality"] = modality
        map_df["Cell"] = cell_type
        map_df["time"] = time_label
        map_df["timepoint"] = timepoint

        return map_df, fr

    def prepare_compound_consensus(
        self,
        profiles: pd.DataFrame,
        replicable_ids: Sequence[str],
    ) -> pd.DataFrame:
        """Prepare compound consensus profiles with target annotations."""
        profiles = remove_negcon_wells(profiles)
        consensus = compute_consensus(profiles, self.config.sample_col)
        consensus = filter_replicable(consensus, replicable_ids)

        # Add target annotations
        consensus = consensus.merge(self.target_annotations, on=self.config.sample_col, how="left")
        consensus[self.config.target_col] = consensus["Metadata_target_list"].str.split("|")
        consensus = consensus.drop(columns=["Metadata_target_list"])

        return consensus

    def prepare_genetic_consensus(
        self,
        profiles: pd.DataFrame,
        replicable_ids: Sequence[str],
    ) -> pd.DataFrame:
        """Prepare genetic perturbation consensus profiles."""
        profiles = remove_negcon_wells(profiles)
        profiles[self.config.target_col] = profiles["Metadata_gene"]

        consensus = compute_consensus(profiles, self.config.sample_col)
        consensus = filter_replicable(consensus, replicable_ids)

        return consensus

    def run_full_evaluation(self) -> EvaluationResults:
        """Run complete benchmark evaluation pipeline.

        Evaluates:
        - Replicability mAP for all modalities
        - Within-modality matching (compound-compound, CRISPR-CRISPR)
        - Cross-modality matching (compound-gene)

        Returns:
            EvaluationResults containing all computed metrics.
        """
        replicability_maps = []
        replicability_frs = []
        matching_maps = []
        matching_frs = []
        cross_modality_maps = []
        cross_modality_frs = []

        for cell_type in self.experiment_df.Cell_type.unique():
            cell_df = self.experiment_df.query(f"Cell_type=='{cell_type}'")

            # Process compounds
            compound_df = cell_df.query("Perturbation=='compound'")
            for timepoint in compound_df.Time.unique():
                print(f"Processing compound_{cell_type}_{timepoint}h")

                profiles = self.load_modality_profiles(cell_type, "compound", timepoint)

                # Replicability
                map_df, fr = self.evaluate_modality_replicability(
                    profiles, "compound", cell_type, timepoint
                )
                replicability_maps.append(map_df)
                replicability_frs.append(
                    self._create_fr_record("compound", cell_type, timepoint, fr, "replicability")
                )

                # Get replicable compounds for matching
                replicable = map_df.query("above_q_threshold==True")[
                    self.config.sample_col
                ].tolist()
                compound_consensus = self.prepare_compound_consensus(profiles, replicable)

                # Compound-compound matching
                result = evaluate_matching(
                    compound_consensus,
                    target_col=self.config.target_col,
                    null_size=self.config.null_size,
                    batch_size=self.config.batch_size,
                    copairs_mode=self.config.copairs_mode,
                )
                match_map = compute_map(
                    result,
                    [self.config.target_col],
                    self.config.q_threshold,
                    null_size=self.config.null_size,
                    copairs_mode=self.config.copairs_mode,
                )
                match_fr = compute_fraction_retrieved(match_map)

                time_label = get_timepoint_label("compound", timepoint)
                match_map["Description"] = f"compound_{cell_type}_{time_label}"
                match_map["Modality"] = "compound"
                match_map["Cell"] = cell_type
                match_map["time"] = time_label

                matching_maps.append(match_map)
                matching_frs.append(
                    self._create_fr_record("compound", cell_type, timepoint, match_fr, "matching")
                )

                # Process genetic perturbations for cross-modality matching
                genetic_df = cell_df.query("Perturbation!='compound'")
                for genetic_pert in genetic_df.Perturbation.unique():
                    pert_df = genetic_df.query(f"Perturbation=='{genetic_pert}'")

                    for gen_timepoint in pert_df.Time.unique():
                        print(f"Processing {genetic_pert}_{cell_type}_{gen_timepoint}h")

                        gen_profiles = self.load_modality_profiles(
                            cell_type, genetic_pert, gen_timepoint
                        )

                        # Replicability (if not already computed)
                        gen_map, gen_fr = self.evaluate_modality_replicability(
                            gen_profiles, genetic_pert, cell_type, gen_timepoint
                        )
                        replicability_maps.append(gen_map)
                        replicability_frs.append(
                            self._create_fr_record(
                                genetic_pert,
                                cell_type,
                                gen_timepoint,
                                gen_fr,
                                "replicability",
                            )
                        )

                        # Get replicable genes
                        gen_replicable = gen_map.query("above_q_threshold==True")[
                            self.config.sample_col
                        ].tolist()
                        gen_consensus = self.prepare_genetic_consensus(gen_profiles, gen_replicable)

                        # CRISPR-CRISPR matching (sister guides)
                        if genetic_pert == "crispr":
                            # Filter to genes with multiple guides
                            guide_counts = gen_consensus["Metadata_gene"].value_counts()
                            multi_guide_genes = guide_counts[guide_counts > 1].index.tolist()
                            crispr_matching = gen_consensus.loc[
                                gen_consensus["Metadata_gene"].isin(multi_guide_genes)
                            ].reset_index(drop=True)

                            if len(crispr_matching) > 0:
                                result = evaluate_matching(
                                    crispr_matching,
                                    target_col=self.config.target_col,
                                    use_abs=False,
                                    null_size=self.config.null_size,
                                    batch_size=self.config.batch_size,
                                    copairs_mode=self.config.copairs_mode,
                                )
                                crispr_map = compute_map(
                                    result,
                                    [self.config.target_col],
                                    self.config.q_threshold,
                                    null_size=self.config.null_size,
                                    copairs_mode=self.config.copairs_mode,
                                )
                                crispr_fr = compute_fraction_retrieved(crispr_map)

                                gen_time = get_timepoint_label(genetic_pert, gen_timepoint)
                                crispr_map["Description"] = f"{genetic_pert}_{cell_type}_{gen_time}"
                                crispr_map["Modality"] = genetic_pert
                                crispr_map["Cell"] = cell_type
                                crispr_map["time"] = gen_time

                                matching_maps.append(crispr_map)
                                matching_frs.append(
                                    self._create_fr_record(
                                        genetic_pert,
                                        cell_type,
                                        gen_timepoint,
                                        crispr_fr,
                                        "matching",
                                    )
                                )

                        # Cross-modality matching
                        # Filter out rows with NaN targets
                        gen_with_targets = gen_consensus.dropna(subset=[self.config.target_col])
                        if gen_with_targets.empty:
                            continue

                        perturbed_genes = set(gen_with_targets[self.config.target_col])
                        compound_filtered = self._filter_to_common_targets(
                            compound_consensus, perturbed_genes
                        )

                        if len(compound_filtered) > 0 and len(gen_with_targets) > 0:
                            # Ensure gen_with_targets has list-type targets for multilabel matching
                            gen_for_matching = gen_with_targets.copy()
                            gen_for_matching[self.config.target_col] = gen_for_matching[
                                self.config.target_col
                            ].apply(lambda x: [x] if isinstance(x, str) else x)

                            combined = pd.concat(
                                [compound_filtered, gen_for_matching],
                                ignore_index=True,
                                join="inner",
                            )

                            # Drop any remaining NaN targets
                            combined = combined.dropna(subset=[self.config.target_col])
                            if combined.empty or len(combined) < 2:
                                continue

                            result = evaluate_cross_modality_matching(
                                combined,
                                target_col=self.config.target_col,
                                null_size=self.config.null_size,
                                batch_size=self.config.batch_size,
                                copairs_mode=self.config.copairs_mode,
                            )
                            cross_map = compute_map(
                                result,
                                [self.config.target_col],
                                self.config.q_threshold,
                                null_size=self.config.null_size,
                                copairs_mode=self.config.copairs_mode,
                            )
                            cross_fr = compute_fraction_retrieved(cross_map)

                            comp_time = get_timepoint_label("compound", timepoint)
                            gen_time = get_timepoint_label(genetic_pert, gen_timepoint)
                            desc = (
                                f"compound_{cell_type}_{comp_time}-"
                                f"{genetic_pert}_{cell_type}_{gen_time}"
                            )

                            cross_map["Description"] = desc
                            cross_map["Modality1"] = f"compound_{comp_time}"
                            cross_map["Modality2"] = f"{genetic_pert}_{gen_time}"
                            cross_map["Cell"] = cell_type

                            cross_modality_maps.append(cross_map)
                            cross_modality_frs.append(
                                {
                                    "Description": desc,
                                    "Modality1": f"compound_{comp_time}",
                                    "Modality2": f"{genetic_pert}_{gen_time}",
                                    "Cell": cell_type,
                                    "fr": cross_fr,
                                }
                            )

        # Aggregate results
        self.results.replicability_map = pd.concat(replicability_maps, ignore_index=True)
        self.results.replicability_fr = pd.DataFrame(replicability_frs)
        self.results.matching_map = pd.concat(matching_maps, ignore_index=True)
        self.results.matching_fr = pd.DataFrame(matching_frs)
        self.results.cross_modality_map = pd.concat(cross_modality_maps, ignore_index=True)
        self.results.cross_modality_fr = pd.DataFrame(cross_modality_frs)

        return self.results

    def _create_fr_record(
        self,
        modality: str,
        cell_type: str,
        timepoint: int,
        fr: float,
        task: str,
    ) -> dict:
        """Create a fraction retrieved record."""
        time_label = get_timepoint_label(modality, timepoint)
        return {
            "Description": f"{modality}_{cell_type}_{time_label}",
            "Modality": modality,
            "Cell": cell_type,
            "time": time_label,
            "timepoint": timepoint,
            "fr": fr,
            "task": task,
        }

    def _filter_to_common_targets(
        self,
        compound_df: pd.DataFrame,
        target_genes: set,
    ) -> pd.DataFrame:
        """Filter compounds to those targeting genes in the target set."""
        if compound_df.empty or not target_genes:
            return pd.DataFrame()

        # Explode multi-label targets and filter
        exploded = compound_df[[self.config.sample_col, self.config.target_col]].copy()
        exploded = exploded.explode(self.config.target_col)
        exploded = exploded[exploded[self.config.target_col].isin(target_genes)]

        if exploded.empty:
            return pd.DataFrame()

        # Re-aggregate targets for remaining compounds
        filtered_ids = exploded[self.config.sample_col].unique()
        filtered = compound_df[compound_df[self.config.sample_col].isin(filtered_ids)].copy()

        if filtered.empty:
            return pd.DataFrame()

        # Update target lists to only include common targets
        target_mapping = (
            exploded.groupby(self.config.sample_col)[self.config.target_col].apply(list).to_dict()
        )
        filtered = filtered.drop(columns=[self.config.target_col])
        filtered[self.config.target_col] = filtered[self.config.sample_col].map(target_mapping)

        return filtered.reset_index(drop=True)
