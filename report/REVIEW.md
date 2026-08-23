# MorphoCLIP Manuscript Review

This review summarizes the current manuscript and the work that remains before a
peer-reviewed submission. It reflects the report as written in the LaTeX sources.

## Overall Assessment

The manuscript presents a clear preliminary study of text-supervised representation
learning for Cell Painting data. The model is described consistently, the retrieval
protocol uses appropriate chance baselines, and the results distinguish between
perturbation retrieval, replicability, target matching, and gene--compound matching.

The strongest result is that MorphoCLIP retrieves held-out perturbations well above
chance in both image-to-text and text-to-image directions. Replicate alignment also
improves agreement between repeated wells across all standard replicability tracks.
The main limitation is that these improvements do not yet produce reliable
gene--compound matching.

The work is suitable as a transparent preprint. Stronger evidence is needed before
making broader claims about biological mechanism or generalization.

## Scientific Review

### Model and Method

- One model represents compounds, CRISPR knockouts, and ORF overexpressions.
- Frozen DINOv3 and BioClinical ModernBERT backbones keep training practical on a
  single consumer GPU.
- The CrossChannelFormer preserves channel identity and combines information across
  the five fluorescence stains.
- The training objective and optional replicate, gene-aware, and plate-correction
  components are described with enough detail to reproduce the experiments.

### Evaluation

- Perturbations are separated across training, validation, and test splits.
- Retrieval is reported at both well and perturbation levels.
- Image-to-text retrieval uses one text candidate per perturbation.
- Analytic chance baselines are reported for each retrieval setting.
- The standard CPJUMP1 benchmark provides separate measurements of replicability,
  within-modality target matching, and cross-modality gene--compound matching.

### Main Findings

| Question | Current evidence |
|---|---|
| Does image--text retrieval work? | Yes. Top-ten retrieval is substantially above chance on validation and test data. |
| Does replicate alignment help? | It improves replicability mAP across all benchmark tracks. |
| Do gene-aware labels improve retrieval? | No consistent improvement is established from one seed. |
| Do plate offsets help? | No consistent retrieval or benchmark improvement is established. |
| Does the model recover gene--compound relationships? | Not reliably; performance remains near zero. |
| Are baseline comparisons decisive? | No. Differences in preprocessing and benchmark randomness prevent a strict ranking. |

The replicability result should be interpreted narrowly because the replicate loss
directly optimizes the relationship measured by that benchmark. Evidence of stronger
biological structure should come from a downstream task that the loss does not optimize.

## Remaining Work

- Repeat the control, replicate-loss, and combined experiments with additional seeds,
  then report means and standard deviations.
- Align the compound target annotations used during training with the broader target
  lists used by the cross-modality benchmark.
- Ablate prompt fields such as SMILES, target genes, and gene-function descriptions.
- Add qualitative nearest-neighbor examples with biological interpretation.
- Evaluate transfer on a larger and more diverse portion of the JUMP dataset.
- Compare baselines under matched preprocessing when practical.

## Final Checks

- Confirm that every table value can be traced to a saved evaluation artifact.
- Keep validation-based checkpoint selection separate from test reporting.
- Describe fraction retrieved as stochastic and replicability mAP as deterministic.
- Avoid interpreting small single-seed differences as reliable improvements.
- Build the PDF and inspect tables, figures, citations, and column flow before release.
- Package only the LaTeX sources, bibliography output, and figures required by the paper.

## Recommendation

Release the current manuscript as a preliminary preprint with the limitations stated
clearly. Prioritize repeated seeds and cross-modality experiments for the next revision,
since these are the main requirements for supporting stronger scientific conclusions.
