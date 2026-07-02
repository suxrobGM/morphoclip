"""
Metadata Analysis for CPJUMP1 External Metadata
================================================
Explores patterns in compounds, CRISPR, and ORF perturbation metadata
with multiple visualizations to guide model design and data selection.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Project paths
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.resolve().parents[2]
_METADATA_DIR = _PROJECT_ROOT / "data" / "metadata" / "external_metadata"
_OUTPUT_DIR = _PROJECT_ROOT / "output" / "metadata_analysis"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load compound, CRISPR, and ORF metadata TSVs."""
    compound_path = _METADATA_DIR / "JUMP-Target-1_compound_metadata_targets.tsv"
    if not compound_path.exists():
        compound_path = _METADATA_DIR / "JUMP-Target-1_compound_metadata.tsv"

    compounds = pd.read_csv(compound_path, sep="\t")
    crispr = pd.read_csv(_METADATA_DIR / "JUMP-Target-1_crispr_metadata.tsv", sep="\t")
    orf = pd.read_csv(_METADATA_DIR / "JUMP-Target-1_orf_metadata.tsv", sep="\t")

    return compounds, crispr, orf


def filter_treatment(compounds: pd.DataFrame, crispr: pd.DataFrame, orf: pd.DataFrame) -> None:
    """Filter to treatment rows only (exclude negcon/control)."""
    compounds.query("control_type != 'negcon'", inplace=True)
    crispr.query("control_type != 'negcon'", inplace=True)
    orf.query("control_type != 'negcon'", inplace=True)


def plot_modality_distribution(
    compounds: pd.DataFrame, crispr: pd.DataFrame, orf: pd.DataFrame
) -> None:
    """Bar chart: count per modality (compound, CRISPR, ORF)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = {
        "Compound": len(compounds),
        "CRISPR": len(crispr),
        "ORF": len(orf),
    }
    bars = ax.bar(counts.keys(), counts.values(), color=["#2ecc71", "#3498db", "#9b59b6"])
    ax.set_ylabel("Count")
    ax.set_title("Perturbation Count by Modality")
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(int(bar.get_height())),
            ha="center",
            fontsize=11,
        )
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "01_modality_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 01_modality_distribution.png")


def plot_top_compound_targets(compounds: pd.DataFrame, top_n: int = 20) -> None:
    """Bar chart: top genes targeted by compounds (primary target)."""
    target_col = "target_list" if "target_list" in compounds.columns else "target"
    target_col = target_col if target_col in compounds.columns else "target"
    primary = compounds[target_col].dropna().astype(str)
    primary = primary[primary.str.strip() != ""]
    primary = primary.str.split("|").str[0].str.strip()
    counts = primary.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax, color="#2ecc71", alpha=0.8)
    ax.set_xlabel("Number of compounds")
    ax.set_ylabel("Primary target gene")
    ax.set_title(f"Top {top_n} Genes Targeted by Compounds")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "02_top_compound_targets.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 02_top_compound_targets.png")


def plot_top_crispr_genes(crispr: pd.DataFrame, top_n: int = 20) -> None:
    """Bar chart: top genes in CRISPR (with guide count)."""
    gene_counts = crispr["gene"].dropna().value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    gene_counts.plot(kind="barh", ax=ax, color="#3498db", alpha=0.8)
    ax.set_xlabel("Number of guides")
    ax.set_ylabel("Gene")
    ax.set_title(f"Top {top_n} Genes in CRISPR (guides per gene)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "03_top_crispr_genes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 03_top_crispr_genes.png")


def plot_top_orf_genes(orf: pd.DataFrame, top_n: int = 20) -> None:
    """Bar chart: top genes in ORF (1 per gene typically)."""
    gene_counts = orf["gene"].dropna().value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    gene_counts.plot(kind="barh", ax=ax, color="#9b59b6", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_ylabel("Gene")
    ax.set_title(f"Top {top_n} Genes in ORF")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "04_top_orf_genes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 04_top_orf_genes.png")


def plot_targets_per_compound(compounds: pd.DataFrame) -> None:
    """Histogram: number of targets per compound (from target_list pipe count)."""
    target_col = "target_list" if "target_list" in compounds.columns else "target"
    target_col = target_col if target_col in compounds.columns else "target"
    counts = compounds[target_col].dropna().astype(str).str.count("\\|") + 1
    counts = counts[counts <= 50]  # cap for display

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        counts, bins=range(1, int(counts.max()) + 2), color="#2ecc71", alpha=0.7, edgecolor="black"
    )
    ax.set_xlabel("Number of targets per compound")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Distribution of Target Count per Compound")
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "05_targets_per_compound.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_targets_per_compound.png")


def plot_smiles_length(compounds: pd.DataFrame) -> None:
    """Histogram: SMILES string length distribution."""
    lengths = compounds["smiles"].dropna().astype(str).str.len()
    lengths = lengths[lengths > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=30, color="#e74c3c", alpha=0.7, edgecolor="black")
    ax.set_xlabel("SMILES length (characters)")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Distribution of SMILES Length")
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "06_smiles_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_smiles_length.png")


def plot_gene_overlap(compounds: pd.DataFrame, crispr: pd.DataFrame, orf: pd.DataFrame) -> None:
    """Venn-style overlap: genes in compounds vs CRISPR vs ORF."""
    target_col = "target_list" if "target_list" in compounds.columns else "target"
    target_col = target_col if target_col in compounds.columns else "target"
    compound_genes = set(
        compounds[target_col].dropna().astype(str).str.split("|").explode().str.strip().str.upper()
    )
    compound_genes = {g for g in compound_genes if g and g != "NAN"}

    crispr_genes = set(crispr["gene"].dropna().astype(str).str.strip().str.upper())
    crispr_genes = {g for g in crispr_genes if g and g != "NAN"}

    orf_genes = set(orf["gene"].dropna().astype(str).str.strip().str.upper())
    orf_genes = {g for g in orf_genes if g and g != "NAN"}

    overlap_cr = len(compound_genes & crispr_genes)
    overlap_co = len(compound_genes & orf_genes)
    overlap_ro = len(crispr_genes & orf_genes)
    overlap_all = len(compound_genes & crispr_genes & orf_genes)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        ["Compound", "CRISPR", "ORF"],
        [len(compound_genes), len(crispr_genes), len(orf_genes)],
        color=["#2ecc71", "#3498db", "#9b59b6"],
        alpha=0.8,
        label="Unique genes",
    )
    ax.set_ylabel("Number of unique genes")
    ax.set_title("Unique Genes per Modality")
    ax.legend()
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "07_gene_counts_per_modality.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_gene_counts_per_modality.png")

    # Overlap table
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    table_data = [
        ["Overlap", "Count"],
        ["Compound ∩ CRISPR", str(overlap_cr)],
        ["Compound ∩ ORF", str(overlap_co)],
        ["CRISPR ∩ ORF", str(overlap_ro)],
        ["All three", str(overlap_all)],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=[0.6, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    ax.set_title("Gene Overlap Between Modalities")
    fig.savefig(_OUTPUT_DIR / "08_gene_overlap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 08_gene_overlap.png")


def plot_crispr_guides_per_gene(crispr: pd.DataFrame) -> None:
    """Histogram: number of CRISPR guides per gene (2 per gene typically)."""
    gene_counts = crispr["gene"].dropna().value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        gene_counts.values,
        bins=range(1, int(gene_counts.max()) + 2),
        color="#3498db",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xlabel("Number of guides per gene")
    ax.set_ylabel("Number of genes")
    ax.set_title("CRISPR Guides per Gene Distribution")
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "09_crispr_guides_per_gene.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 09_crispr_guides_per_gene.png")


def plot_compound_name_length(compounds: pd.DataFrame) -> None:
    """Histogram: compound name (pert_iname) length."""
    name_col = "pert_iname" if "pert_iname" in compounds.columns else "broad_sample"
    lengths = compounds[name_col].dropna().astype(str).str.len()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=25, color="#f39c12", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Compound name length (characters)")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Distribution of Compound Name Length")
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "10_compound_name_length.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 10_compound_name_length.png")


def plot_pubchem_coverage(compounds: pd.DataFrame) -> None:
    """Pie chart: compounds with vs without PubChem ID."""
    if "pubchem_cid" not in compounds.columns:
        return
    has_pubchem = compounds["pubchem_cid"].notna() & (
        compounds["pubchem_cid"].astype(str).str.strip() != ""
    )
    counts = [has_pubchem.sum(), (~has_pubchem).sum()]
    labels = ["With PubChem ID", "Without PubChem ID"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#27ae60", "#c0392b"])
    ax.set_title("Compound PubChem ID Coverage")
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "11_pubchem_coverage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 11_pubchem_coverage.png")


def plot_summary_dashboard(
    compounds: pd.DataFrame, crispr: pd.DataFrame, orf: pd.DataFrame
) -> None:
    """2x2 summary dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Modality counts
    counts = [len(compounds), len(crispr), len(orf)]
    axes[0, 0].bar(["Compound", "CRISPR", "ORF"], counts, color=["#2ecc71", "#3498db", "#9b59b6"])
    axes[0, 0].set_title("Perturbation Count by Modality")
    axes[0, 0].set_ylabel("Count")

    # 2. SMILES length (compounds)
    lengths = compounds["smiles"].dropna().astype(str).str.len()
    lengths = lengths[lengths > 0]
    axes[0, 1].hist(lengths, bins=25, color="#e74c3c", alpha=0.7)
    axes[0, 1].set_title("SMILES Length (compounds)")
    axes[0, 1].set_xlabel("Length")

    # 3. Targets per compound
    target_col = "target_list" if "target_list" in compounds.columns else "target"
    n_targets = compounds[target_col].dropna().astype(str).str.count("\\|") + 1
    n_targets = n_targets[n_targets <= 30]
    axes[1, 0].hist(n_targets, bins=range(1, int(n_targets.max()) + 2), color="#2ecc71", alpha=0.7)
    axes[1, 0].set_title("Targets per Compound")
    axes[1, 0].set_xlabel("Number of targets")

    # 4. CRISPR guides per gene
    gene_counts = crispr["gene"].dropna().value_counts()
    axes[1, 1].hist(
        gene_counts.values, bins=range(1, int(gene_counts.max()) + 2), color="#3498db", alpha=0.7
    )
    axes[1, 1].set_title("CRISPR Guides per Gene")
    axes[1, 1].set_xlabel("Number of guides")

    plt.suptitle("CPJUMP1 Metadata Summary", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(_OUTPUT_DIR / "12_summary_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: 12_summary_dashboard.png")


def main() -> int:
    """Run full metadata analysis and save plots."""
    print("Metadata Analysis for CPJUMP1")
    print("=" * 50)
    print(f"Metadata dir: {_METADATA_DIR}")
    print(f"Output dir:  {_OUTPUT_DIR}")
    print()

    if not _METADATA_DIR.exists():
        print(f"ERROR: Metadata directory not found: {_METADATA_DIR}")
        return 1

    compounds, crispr, orf = load_metadata()
    filter_treatment(compounds, crispr, orf)

    print(
        f"Loaded: {len(compounds)} compounds, {len(crispr)} CRISPR, {len(orf)} ORF (treatment only)"
    )
    print()

    print("Generating plots...")
    plot_modality_distribution(compounds, crispr, orf)
    plot_top_compound_targets(compounds)
    plot_top_crispr_genes(crispr)
    plot_top_orf_genes(orf)
    plot_targets_per_compound(compounds)
    plot_smiles_length(compounds)
    plot_gene_overlap(compounds, crispr, orf)
    plot_crispr_guides_per_gene(crispr)
    plot_compound_name_length(compounds)
    plot_pubchem_coverage(compounds)
    plot_summary_dashboard(compounds, crispr, orf)

    print()
    print(f"Done. All plots saved to: {_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
