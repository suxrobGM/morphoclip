### CPJUMP1 CellProfiler Baseline (Minimal)

This `baseline/` folder is a **self‑contained, minimal copy** of the CPJUMP1 CellProfiler benchmark used in the paper. It is trimmed to only what you need to **run and inspect the baseline results** (no images, no extra notebooks).

---

### Contents

- `environment.yml` — conda environment for the baseline (`analysis`).
- `1.0.calculate-map-cp.ipynb` — notebook that computes/loads the CellProfiler benchmark metrics.
- `utils.py` — helper functions used by the notebook (loading data, running mAP pipeline, etc.).
- `output/`:
  - `experiment-metadata.tsv` — experiment/plate metadata table.
  - `cellprofiler_replicability_map.csv` / `cellprofiler_replicability_fr.csv`
  - `cellprofiler_matching_map.csv` / `cellprofiler_matching_fr.csv`
  - `cellprofiler_gene_compound_matching_map.csv` / `cellprofiler_gene_compound_matching_fr.csv`
  - `replicability_pvalue.csv`, `matching_pvalue.csv`, `gene_compound_matching_pvalue.csv`
  - `cellprofiler_replicability_orf_*` CSVs (additional ORF replicability summaries).
- `input/`:
  - `JUMP-Target-1_compound_metadata_additional_annotations.tsv` — extra compound target annotations.

All paths inside the notebook (`output/...`, `input/...`) are relative and already point to these files.

---

### How to set up the environment

From the repository root:

```bash
cd baseline
conda env create --file environment.yml    # creates env named 'analysis'
conda activate analysis
python -c "import pandas, utils; print('baseline ok')"
```

If you already created `analysis` earlier, you can skip `conda env create` and just:

```bash
cd baseline
conda activate analysis
```

---

### How to run the baseline notebook

1. **Activate the env**:
   ```bash
   cd baseline
   conda activate analysis
   ```
2. **Open the notebook** in your editor (or via Jupyter) and select the `analysis` kernel:
   - File: `1.0.calculate-map-cp.ipynb`
3. **Run all cells**:
   - If you **don’t have the large `profiles/` data**, the notebook will detect that the profile CSVs are Git LFS pointers and will automatically:
     - Skip recomputing everything, and
     - **Load the precomputed benchmark results** from the CSVs in `output/`.

The notebook will then show/plot the CellProfiler baseline metrics (replicability, matching, gene–compound matching) using only the small files shipped here.

---

### Where to find the baseline numbers

- **Primary metrics**:
  - `output/cellprofiler_replicability_map.csv`
  - `output/cellprofiler_replicability_fr.csv`
  - `output/cellprofiler_matching_map.csv`
  - `output/cellprofiler_matching_fr.csv`
  - `output/cellprofiler_gene_compound_matching_map.csv`
  - `output/cellprofiler_gene_compound_matching_fr.csv`

These are the reference scores you should compare your own models/feature sets against.

---

### What is *not* included here

This folder intentionally omits everything that is **not needed to run the baseline**:

- No raw images
- No profile generation pipelines
- No older/deep learning notebooks
- No manuscript or figure‑generation notebooks

If you want to customize or extend the full project, use the original repository structure. If you only care about **“run the official CellProfiler baseline and get the numbers”**, use **only this `baseline/` folder**.

