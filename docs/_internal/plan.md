# Project Plan

**MorphoCLIP — Development Timeline & Task Breakdown**

**Team:** Shubham Gajjar (Lead), Rongfei Jin, Sukhrobbek Ilyosbekov

## Progress Tracker

| Phase                          | Status         | Dates       |
| ------------------------------ | -------------- | ----------- |
| Phase 1: Baseline & EDA        | ✅ Complete    | Weeks 1–2   |
| Phase 2: Data Pipeline         | 🔄 Working     | Weeks 3–4   |
| Phase 3: Model Development     | 🔄 Working     | Weeks 5–8   |
| Phase 4: Evaluation & Ablation | 🔲 Not started | Weeks 9–10  |
| Phase 5: Analysis & Writing    | 🔲 Not started | Weeks 11–12 |

## Phase 1: Baseline Reproduction & EDA ✅

**Goal:** Understand the dataset and establish baseline numbers to beat.

### Completed Tasks

- Clone CPJUMP1 benchmark repository
- Run benchmark/ notebook from Chandrasekaran et al. (2024)
- Generate baseline CellProfiler results:
  - `cellprofiler_replicability_fr.csv` / `_map.csv` — perturbation replicability
  - `cellprofiler_matching_fr.csv` / `_map.csv` — sister perturbation matching
  - `cellprofiler_gene_compound_matching_fr.csv` / `_map.csv` — cross-modal matching
- EDA on benchmark outputs (distributions, cell line comparisons, timepoint effects)
- Literature review (see `docs/literature-review.md` or `literature-review.md`)

### Key Findings

- Confirmed baseline: ~5–25% compound matching, ~7–17% CRISPR matching
- Long timepoints outperform short timepoints
- Gene–compound matching is the hardest task
- mAP distributions are heavily right-skewed (most perturbations near zero)

## Phase 2: Data Pipeline (Weeks 3–4)

**Goal:** Set up efficient data loading for images and metadata.

**Owner:** TBD

### Tasks

- Set up jump_download as submodule in repo
- Configure selective download (source_4, COMPOUND + CRISPR, U2OS, long timepoint)
- Test download with small subset (samples_per_well=1)
- Full download of working subset (~50–150 GB)
- Build metadata linkage: well → plate → perturbation → gene target → SMILES
- Download pre-extracted embeddings:
  - DeepProfiler embeddings from CPJUMP1 repo
  - OpenPhenom embeddings (if available publicly)
- Create PyTorch Dataset class:
  - Load 5-channel images per well
  - Generate text descriptions per perturbation
  - Handle train/val/test splits (use provided splits from repo)
- Implement data augmentations (if using raw images):
  - Per-channel normalization (plate-level median/MAD)
  - Random crops, flips
  - Illumination correction

### Deliverables

- Working DataLoader that yields `(image_tensor, text_string, metadata)` tuples
- Data exploration notebook showing sample images + metadata

## Phase 3: Model Development (Weeks 5–8)

**Goal:** Implement and train MorphoCLIP.

### Week 5–6: Core Architecture

**Owner:** TBD

- Implement image encoder branch:
  - Option A: MLP projection head on top of pre-extracted embeddings
  - Option B: ViT backbone with channel adapter for 5-channel input
  - Attention-based pooling over multiple cells per well
- Implement text encoder branch:
  - Load pre-trained PubMedBERT / SciBERT
  - Text prompt template construction
  - Projection head to shared embedding dimension
- Implement contrastive loss:
  - InfoNCE (baseline)
  - CWCL (CellCLIP's approach)
  - Experiment with SigLip loss
- Training loop with:
  - Mixed precision (fp16)
  - Gradient accumulation
  - Cosine learning rate schedule
  - Logging (WandB or TensorBoard)

### Week 7–8: Training & Iteration

**Owner:** TBD

- Train on pre-extracted embeddings first (fast iteration, 1–2 GPUs)
- Hyperparameter search:
  - Learning rate: [1e-4, 3e-4, 5e-4, 1e-3]
  - Batch size: [256, 512, 1024]
  - Embedding dimension: [256, 512]
  - Loss function: [InfoNCE, CWCL, SigLip]
- Text prompt ablation:
  - Minimal: just perturbation name
  - Standard: name + SMILES/gene
  - Enriched: + pathway info, protein function, DrugBank description
- If time permits: train with raw images (fine-tune ViT backbone)
- Checkpoint best models

### Deliverables

- Trained MorphoCLIP model(s)
- Training curves and hyperparameter comparison plots
- WandB/TensorBoard logs

## Phase 4: Evaluation & Ablation (Weeks 9–10)

**Goal:** Rigorously evaluate MorphoCLIP and understand what works.

**Owner:** TBD

### Evaluation Tasks

- Run CPJUMP1 benchmark evaluation:
  - Replicability (mAP) — compare to CellProfiler baseline
  - Sister perturbation matching (mAP)
  - Gene–compound matching (mAP)
- Generate result CSVs in same format as baseline (for direct comparison)
- If time permits: RxRx3-core gene–gene recovery benchmark (CORUM, HuMAP, Reactome, SIGNOR, StringDB)

### Ablation Study

- Image encoder: pre-extracted embeddings vs. fine-tuned ViT
- Text encoder: frozen vs. fine-tuned PubMedBERT
- Text prompts: minimal vs. standard vs. enriched
- Loss function: InfoNCE vs. CWCL vs. SigLip
- Channel handling: stacked 5-ch vs. CrossChannelFormer-style
- Pooling: mean vs. attention-based
- With vs. without cross-well alignment

### Deliverables

- Comparison table: MorphoCLIP vs. CellProfiler vs. CellCLIP (from paper)
- Ablation table showing contribution of each component
- Per-perturbation analysis: which perturbations improved most?

## Phase 5: Analysis & Writing (Weeks 11–12)

**Goal:** Synthesize results and prepare deliverables.

**Owner:** TBD

### Tasks

- Visualizations:
  - UMAP of learned embedding space (colored by gene target, perturbation type)
  - Attention weight analysis: which cells are most informative?
  - Bar charts: MorphoCLIP vs. baselines across all tasks
  - Per-gene improvement scatter plot
- Write final report / paper draft
- Prepare presentation slides
- Clean up codebase:
  - Docstrings and type hints
  - Reproducibility: random seeds, config files
  - Update README with final results
  - Release model checkpoints (if results are strong)

### Deliverables

- Final report / paper draft
- Presentation slides
- Clean GitHub repository with documentation

## Task Ownership Matrix

| Task Area              | Shubham | Rongfei | Sukhrobbek |
| ---------------------- | ------- | ------- | ---------- |
| Baseline & EDA         | Lead ✅ |         |            |
| Data pipeline          |         |         |            |
| Image encoder          |         |         |            |
| Text encoder           |         |         |            |
| Training loop          |         |         |            |
| Evaluation             |         |         |            |
| Ablation study         |         |         |            |
| Writing & presentation |         |         |            |

_Note: Fill in ownership as tasks are assigned. Each member should own at least 2 major areas._

## Key Dependencies & Milestones

```
Week 1–2   ✅ Baseline reproduced, EDA complete
    │
Week 3     Data pipeline working (can load images + metadata)
    │
Week 4     Pre-extracted embeddings downloaded & verified
    │
Week 5     Core MorphoCLIP architecture implemented
    │
Week 6     First training run complete (on embeddings)
    │
Week 8     Best model selected, hyperparams tuned
    │         ← CRITICAL MILESTONE: do we beat the baseline?
Week 10    Full evaluation + ablation complete
    │
Week 12    Report + presentation ready
```

## Compute Plan

| Phase                | Resource             | Estimated Time    | Estimated Cost     |
| -------------------- | -------------------- | ----------------- | ------------------ |
| Phase 1              | CPU / local          | Done              | Free               |
| Phase 2              | CPU + S3 download    | 1–2 days          | Free (public data) |
| Phase 3 (embeddings) | 1–2 GPUs (V100/A100) | 2–3 days training | ~$50–100 (cloud)   |
| Phase 3 (raw images) | 4 GPUs (A100)        | 3–5 days training | ~$200–500 (cloud)  |
| Phase 4              | 1 GPU                | 1 day             | ~$20               |

Consider using university cluster / lab GPUs to reduce cost. Pre-extracted embeddings approach is feasible on a single GPU.

## Risk Register

| #   | Risk                           | Likelihood | Impact | Mitigation                                            | Status |
| --- | ------------------------------ | ---------- | ------ | ----------------------------------------------------- | ------ |
| 1   | Data download too slow / fails | Low        | Medium | Start early; use profiles first                       | Open   |
| 2   | Not enough GPU compute         | Medium     | High   | Use pre-extracted embeddings; apply for cloud credits | Open   |
| 3   | Model doesn't beat baseline    | Medium     | High   | Reproduce CellCLIP first; incremental changes         | Open   |
| 4   | Batch effects dominate         | Medium     | High   | Cross-well alignment; per-plate normalization         | Open   |
| 5   | Team member unavailability     | Low        | Medium | Clear ownership; overlap on critical tasks            | Open   |
