# MorphoCLIP Manuscript Review

> **Superseded in part — read [Addendum: corrected evaluation](#addendum-corrected-evaluation-2026-08-07) first.**
> Section A below concluded that image→text retrieval sits at or below chance. That
> conclusion was itself an evaluation artifact. The model is roughly 4x above chance in
> *both* directions once the candidate pool is de-duplicated.


Reviewer pass for the arXiv preprint / journal track. This document records the
quality rating, the findings behind the edits applied to the `.tex` sources, and a
checklist of the evidence still needed before a peer-reviewed submission. All numbers
below were verified against the code and the reported run
(`output/morphoclip_runs/ccf_preload/`), whose metrics match the paper's headline.

## Rating

| Dimension | Preprint | Journal | Note |
|---|---|---|---|
| Writing / clarity | 8/10 | 8/10 | Clean narrative; a few AI-tone tells, now trimmed |
| Method–code fidelity | 4/10 | 4/10 | CCF, prompt templates, param count, split, batch size were all wrong in the text |
| Novelty framing | 6/10 | 5/10 | CWA claimed as delivered but is off in the reported run |
| Empirical support | 3/10 | 2/10 | Inflated baseline; ablations unrun; single seed; well-level protocol |
| Baseline rigor | 3/10 | 3/10 | Non-comparable baselines; random reference was ~100x too low |
| Formatting / packaging | 6/10 | 6/10 | Table I column bug; unused figures; build artifacts in tree |
| **Overall** | **~5/10** | **~3/10** | Honest preliminary report after the corrections below; journal needs the evidence track |

The prose is genuinely good. The problems are almost entirely **factual**: the manuscript
described a model and an evaluation that differ from the code that produced the numbers.

## A. Results integrity — the load-bearing finding

The reported run ranks retrieval over **individual wells**, not perturbation profiles
([`src/morphoclip/training/evaluate.py`](../src/morphoclip/training/evaluate.py) L55–83).
The validation split is **2,220 wells drawn from 98 perturbations** (42 compound / 34 CRISPR
/ 22 ORF), i.e. ~24 replicate wells per perturbation, each replicate sharing an identical
text embedding. This was reconstructed exactly from metadata (reproduces N = 2,220) after the
feature cache was found purged.

Two consequences:

1. **"817 candidate perturbations" is wrong.** 817 is the total number of cached text
   embeddings for the whole dataset, not the validation pool. The pool is 98 perturbations /
   2,220 wells.
2. **The random baseline was ~100x too low.** The paper compared results to a single-positive
   baseline over 817 candidates (R@1 ≈ 0.12%, median ≈ 408). With ~24 positives per query over
   2,220 wells, the correct like-for-like random baseline is:

   | | Paper's random | **True random** | MorphoCLIP |
   |---|---|---|---|
   | R@1 | 0.12% | **1.09%** | t→i 3.51 / i→t 5.72 |
   | R@5 | (0.61%) | **5.33%** | t→i 14.86 / i→t 6.71 |
   | R@10 | ~0.12% | **10.37%** | t→i 24.32 / i→t **7.07** |
   | median rank | 408 | **~64** | t→i 28 / i→t 289 |

   So text→image R@10 (24.3%) is **~2.3× random, not "two orders of magnitude"**, and
   **image→text is at or below chance** (R@10 7.07% < 10.37%; median 289 ≫ 64). The
   "median rank 28 → top 3.4%" figure also used 817; against 2,220 it is top 1.3%.

**Root cause:** the manuscript describes perturbation-level retrieval over 817 candidates, but
the code implements well-level retrieval over 2,220 wells with duplicated text vectors. The fix
adopted in the text is an honest reframe against the true baseline; the *proper* fix is to
re-run evaluation at the perturbation-aggregated level (standard CPJUMP1 protocol), which should
also lift image→text out of the noise floor. **This is the top pre-journal action item.**

## B. Method–code fidelity (corrected in the text)

- **Trainable parameters: ~8M → ~14M.** Summed directly from the checkpoint:
  image encoder 13.40M (CCF 12.60M + projection 0.79M) + text projection 0.66M + logit scale =
  **14.05M**. Corrected in the abstract, intro, related-work table, methods, results table.
- **CrossChannelFormer aggregation.** The paper's `MeanPool(CCF(...))` is wrong. The code
  ([`cross_channel_former.py`](../src/morphoclip/models/cross_channel_former.py)) L2-normalizes
  each channel token, adds a **learnable channel-type embedding**, prepends a **learnable
  aggregation (CLS) token**, and reads out that token's transformer output. Mean pooling happens
  only at the **site→well** step. Methods rewritten accordingly. *(The architecture figure PDF
  may still depict mean pooling; regenerate `figures/architecture.pdf` if so.)*
- **Prompt templates.** The templates quoted in the paper do not match
  [`prompts.py`](../src/morphoclip/models/prompts.py). In particular the CRISPR
  "GO terms: {biological\_process\_terms}" field **does not exist**; the real templates name the
  cell line and a "Perturbation modality" tag. Replaced with the actual templates; the
  annotation-source sentence now says UniProt (not GO).
- **Batch size: 512 → 256.** The reported checkpoint's config is `batch_size: 256`.
- **Split description.** The paper's default "CRISPR/ORF → train, compounds → val/test" is not
  what ran. The `pert_type` strategy hashes each `broad_sample` into an 80/10/10 train/val/test
  split with all modalities in every split ([`strategies.py`](../src/morphoclip/splits/strategies.py)).
  Corrected in the experiments section.
- **Checkpoint selection.** Best-on-validation is epoch 9 of the 100-epoch schedule, now stated.

## C. Claim honesty

- **CWA is off in the reported run** (`use_cwa: false`, [`configs/train/base.yaml`](../configs/train/base.yaml)).
  Everywhere CWA was presented as a delivered contribution it is now "optional / evaluated in
  ablation," and Table I marks it "CWA (opt.)".
- **Validation, single seed.** Stated plainly; the held-out test split is future work.
- **Baseline comparisons** are framed as indicative; the image→text "parity" with CellCLIP is
  now qualified because both sit near this setup's random baseline.

## D. Formatting

- **Table I column bug.** `\begin{tabular}{@{}lccccc@{}}` declared 6 columns for 5 headers.
  Fixed by adding the **Text** column the narrative referenced (order: Method | Image Encoder |
  Text | Pert. Types | Batch Corr. | Params).
- **Feature figures.** `figures/variance_decomp.png`, `channel_similarity.png`,
  `pca_channels.png` existed but were never included; they are now referenced in the feature
  analysis (Figures 3–5). The contradictory 0.82 vs 0.75 cross-channel-similarity numbers were
  reconciled (the redundant 0.82 sanity-check sentence in methods was removed).
- **arXiv packaging.** Submit only compile inputs (`.tex`, `.bib`/`.bbl`, used figures, IEEEtran
  class). Exclude the build artifacts currently in `report/` (`.aux .log .out .fls .fdb_latexmk
  .synctex.gz .blg`) and the stale `main.pdf` before uploading.

## E. Tone / bloat trimmed

Removed the "deceptively simple question" opener, "leaves something on the table", "attacks the
problem from another angle", "headline number", "the contribution we want to highlight",
"opens up", and "decorative"; cut the repeated "~8M vs 1.48B" line down to the abstract, related
work, and results; stated the "three capabilities" framing once. Contribution verbs made
cautious ("we present / we evaluate / preliminary results suggest").

## Pre-journal checklist

- [x] **Re-evaluate at the perturbation level** — done; see the addendum below.
- [ ] Run the **2³ factorial ablation** (text × CWA × genes) plus the mean-pool and
      prompt-richness variants on one locked split. *(Staged 4-run campaign in progress.)*
- [ ] **≥3 seeds** (42, 1337, 2024); report mean ± sd for R@1/5/10 and median rank.
- [ ] Report the headline on the **held-out test split** after validation-based model selection.
- [ ] Same-data **head-to-head vs CWA-MSN** (CORUM / HuMAP / Reactome) and CellCLIP.
- [ ] **Enable CWA and measure it**, or drop it from the contribution list. *(In the campaign.)*
- [x] ~~Investigate why **image→text is below chance**~~ — it never was; see the addendum.
- [ ] Add qualitative **nearest-neighbor examples** with biological interpretation.
- [ ] Regenerate the **architecture figure** if it shows channel mean-pooling.

---

## Addendum: corrected evaluation (2026-08-07)

### The image→text result was an evaluation artifact, not a model failure

`compute_retrieval_metrics` ranked each image query against all 2,220 well-texts. But
replicate wells share an *identical* text embedding, so that pool held only 98 distinct
vectors, each duplicated ~24 times. A model scores identical vectors identically, so all
copies of a text land at consecutive ranks — which caps R@5/R@10 near R@1 and makes the
shuffled-well random baseline inapplicable to that direction. The random baseline
interleaved the copies; the model could not.

Retrieval now ranks the 98 **unique** texts, and reports analytic baselines alongside every
direction. Same checkpoint (`ccf_preload`, epoch 9), same split, corrected metric:

| Direction | R@1 | R@5 | R@10 | Median rank | Random R@10 |
|---|---|---|---|---|---|
| image→text (well-level) | 5.90% | 23.33% | **39.41%** | 14 / 98 | 10.20% |
| text→image (well-level) | 4.08% | 15.31% | 23.47% | 30 | 9.74% |
| image→text (perturbation) | 4.08% | 20.41% | **38.78%** | 13 / 98 | 10.20% |
| text→image (perturbation) | 5.10% | 18.37% | **39.80%** | 13 / 98 | 10.20% |

Two conclusions follow, both favorable and both requiring manuscript changes:

1. **The model is ~4x above chance in both directions**, not "at or below chance" in one.
   The observed median rank of 14 matches the ~13 predicted from the clumping arithmetic,
   which confirms the artifact diagnosis.
2. **The retrieval asymmetry does not exist.** At the perturbation level the two directions
   are within a point of each other (38.8% vs 39.8%). Section "Retrieval Asymmetry" in
   `results.tex` explains a measurement bug and should be deleted, not rewritten. Its stated
   premise — "the positive mask is symmetric, so the random baseline is identical for both" —
   is exactly the error: the *mask* is symmetric, candidate *multiplicity* is not.

### First CPJUMP1 benchmark numbers (standard protocol)

`morphoclip export-profiles` writes benchmark-layout profiles, so the same stable harness used
for CellProfiler now runs on MorphoCLIP embeddings. Fraction retrieved, 40 benchmark-eligible
plates:

| Task | A549 | U2OS |
|---|---|---|
| Compound replicability (short / long) | 0.35 / 0.51 | 0.30 / 0.37 |
| CRISPR replicability (short / long) | 0.17 / 0.18 | 0.03 / 0.17 |
| ORF replicability | ~0.01 | ~0.02 |
| **Gene–compound (cross-modality)** | **~0.00** | **~0.00** |

Compound replicability is competitive with CellProfiler's 5–25% range. ORF and cross-modality
matching are the real weaknesses and should be reported as such — cross-modality is the task
the method is ultimately pitched at.

### Staged ablation campaign (seed 42, 30-epoch schedule, early stop patience 8)

Four runs off one shared config (`configs/train/ablation.yaml`), variants supplied as CLI
overrides. All use the perturbation-aware batch sampler. Validation split, corrected metrics:

| Run | Change | pert i2t R@10 | pert t2i R@10 | Epochs (best) |
|---|---|---|---|---|
| `abl_repro` | control | 0.378 | 0.378 | 19 (11) |
| `abl_soft` | `target_weight=0.6` | 0.378 | **0.429** | 21 (13) |
| `abl_imgimg` | `lambda_img=0.3` | **0.449** | 0.418 | 15 (7) |
| `abl_cwa` | `use_cwa=true` | 0.153 | 0.102 | 12 (4) |
| *random* | — | 0.102 | 0.102 | — |

Three findings, in order of importance:

1. **The replicate-alignment image–image loss is the win: +18.9% over control** (0.378 → 0.449),
   and it converges faster (best epoch 7 vs 11). This supports the diagnosis that the binding
   constraint was a *missing training signal* — nothing previously pulled replicate wells of the
   same perturbation together across plates — rather than insufficient aggregator capacity.
2. **Gene-aware soft labels help one direction only**: text→image improves 0.378 → 0.429
   (+13.5%), image→text is unchanged. Consistent with soft positives sharpening the text-anchor
   geometry without adding information to the image side.
3. **CWA is catastrophic, and this is a claim-level problem.** Enabling cross-well alignment
   drops perturbation-level retrieval to 0.153 against a 0.102 random floor — the model barely
   beats chance, and it early-stops at epoch 12 with its best at epoch 4. I checked whether this
   was an eval-time artifact by re-evaluating the same checkpoint with CWA disabled at inference:
   it scored **0.122**, no better. So the training itself fails.

   A plausible mechanism: in CPJUMP1 a plate is nearly synonymous with a *condition* (one
   modality, one cell line, one timepoint). Subtracting the per-plate mean therefore removes
   modality and cell-line signal — exactly what the text prompts encode ("Perturbation modality:
   CRISPR knockout", the cell line name). CWA is designed for a unimodal setting like CWA-MSN's;
   in a text-aligned model it deletes the very axes the text describes.

   **Manuscript consequence:** CWA cannot remain in the contribution list on the strength of the
   design alone. Either report this negative result explicitly, or drop the component. The
   current framing ("optional / evaluated in ablation") is no longer sufficient now that the
   ablation exists and is strongly negative.

### The winning run on the standard benchmark — and why the gain needs a caveat

`abl_imgimg` exported and benchmarked against the `ccf_preload` baseline, fraction retrieved:

| Track | Baseline | `abl_imgimg` | Δ |
|---|---|---|---|
| Replicability (mean of 12) | 0.178 | **0.248** | **+0.070** |
| ↳ compound A549 long | 0.513 | 0.650 | +0.137 |
| ↳ CRISPR A549 long | 0.183 | 0.376 | +0.193 |
| ↳ ORF (all four) | ~0.01 | ~0.03 | ~+0.02 |
| Target matching (mean of 8) | 0.339 | 0.262 | −0.077 |
| Gene–compound cross-modality | 0.005 | 0.000 | −0.005 |

Replicability improves on **every one of the twelve** tracks (+39% relative). But this must be
reported with the caveat that it is partly definitional: the replicate-alignment loss directly
optimizes similarity between replicate wells, and replicability fraction-retrieved measures
exactly replicate retrieval. Claiming it as an independent win would be teaching to the test.

The honest reading:

- **Independent evidence exists** — perturbation-level text↔image retrieval also improved
  (+18.9%), and that metric is not what the loss optimizes.
- **Target matching did not improve** (0.339 → 0.262 mean), though the per-track numbers are
  noisy: `crispr_U2OS_short` moves 1.000 → 0.667 on what is almost certainly a handful of
  perturbations. These tracks need seed replication before any claim rests on them.
- **Cross-modality gene–compound matching remains at zero.** This is the task the method is
  ultimately pitched at, and none of the three training-signal changes moved it. It should be
  stated plainly as an open failure rather than omitted.

Better replicate alignment therefore buys a tighter within-perturbation cluster without yet
buying the cross-modality structure that would make compound→gene matching work.

### Two infrastructure findings worth a methods sentence

- **Feature cache was 44x oversized.** `torch.save` on a bare slice serializes the slice's
  entire backing storage, so each 20 KiB `(5, 1024)` site tensor was written as a ~960 KiB
  file: 168 GiB instead of 3.8 GiB. Fixed in the extractor; `morphoclip features repack`
  rewrites existing caches. The paper's "~3 GB per plate / ~153 GB total" figures describe the
  bug, not the data — the correct figure is ~75 MB per plate.
- **Soft-positive gene source.** Compound annotations carry both `target` (primary gene, 160
  distinct) and `target_list` (includes off-targets, 758 distinct). Training soft labels use
  `target`; the CPJUMP1 benchmark's cross-modality matching uses `target_list`. That mismatch
  is defensible — `target_list` would link any two calcium-channel blockers through a 26-gene
  list — but it is a deliberate choice worth stating, and a candidate ablation given that
  cross-modality retrieval is currently at zero.
