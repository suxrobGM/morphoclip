# Benchmark Pipeline

This document explains the benchmark evaluation process for perturbation embedding quality assessment. The pipeline measures how well learned embeddings capture biological similarity using information retrieval metrics.

## Overview

The benchmark evaluates embeddings on three tasks:

1. **Replicability** - Can the model retrieve replicate samples of the same perturbation?
2. **Target Matching** - Can the model retrieve perturbations that affect the same biological target?
3. **Cross-Modality Matching** - Can the model match chemical perturbations to genetic perturbations of the same target?

## Data Structure

### Input: Feature Profiles

Each sample is represented as a high-dimensional feature vector:

```
Sample = [Metadata columns..., Feature columns...]
        |__________________|  |__________________|
         ~20 columns           ~1000+ features
```

- **Metadata columns**: Prefixed with `Metadata_` (e.g., `Metadata_broad_sample`, `Metadata_Plate`)
- **Feature columns**: Numerical embeddings/features used for similarity computation

### Key Metadata Fields

| Field                   | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `Metadata_broad_sample` | Unique perturbation identifier                 |
| `Metadata_Plate`        | Plate identifier (batch effect source)         |
| `Metadata_control_type` | `negcon` for negative controls, else treated   |
| `Metadata_gene`         | Target gene (for genetic perturbations)        |
| `Metadata_modality`     | Perturbation type: `compound`, `crispr`, `orf` |

## Evaluation Metrics

### Mean Average Precision (mAP)

The primary metric is **mean Average Precision**, a standard information retrieval metric.

For each query sample `q`:

1. Compute cosine similarity to all other samples
2. Rank samples by similarity (descending)
3. Compute Average Precision:

```
AP(q) = (1/R) * Σ(k=1 to n) [Precision@k * rel(k)]

where:
  R = number of relevant (positive) samples
  rel(k) = 1 if sample at rank k is relevant, 0 otherwise
  Precision@k = (# relevant in top k) / k
```

4. Aggregate across all queries:

```
mAP = (1/Q) * Σ(q=1 to Q) AP(q)
```

### Fraction Retrieved

A simpler metric measuring the proportion of perturbations that achieve statistical significance:

```
FR = |{p : q-value(p) < 0.05}| / |P|

where P is the set of all perturbations
```

Q-values are FDR-corrected p-values from a permutation test.

## Task Definitions

### Task 1: Replicability

**Question**: Do replicate measurements of the same perturbation cluster together?

```
Positive pairs:  Same Metadata_broad_sample, different wells
Negative pairs:  Same plate, different perturbation (non-control)
```

This measures technical reproducibility and embedding quality for distinguishing perturbations.

### Task 2: Target Matching

**Question**: Do perturbations affecting the same target cluster together?

```
Positive pairs:  Same target gene/pathway
Negative pairs:  Different targets
```

For compounds with multiple targets, this uses multi-label matching (any overlapping target counts as positive).

**Anti-correlation handling**: Uses absolute cosine similarity because perturbations can have opposite effects on the same target.

### Task 3: Cross-Modality Matching

**Question**: Do chemical and genetic perturbations of the same target match?

```
Positive pairs:  Same target, different modality (compound vs. gene)
Negative pairs:  Different targets OR same modality
```

This is the most challenging task, requiring the embedding to capture functional similarity across fundamentally different perturbation types.

## Pipeline Steps

### Step 1: Data Loading and Preprocessing

```python
# Load normalized feature profiles
profiles = load_plate_data(batch, plate, "normalized_feature_select_negcon_batch.csv.gz")

# Remove empty wells
profiles = profiles.dropna(subset=["Metadata_broad_sample"])

# Add negative control indicator
profiles["Metadata_negcon"] = (profiles["Metadata_control_type"] == "negcon").astype(int)
```

### Step 2: Pair Generation

The `copairs` library generates positive and negative pairs based on matching rules:

```python
from copairs.matching import Matcher

# Define matching rules
pos_sameby = ["Metadata_broad_sample"]  # Must match for positive pairs
pos_diffby = []                          # Must differ for positive pairs
neg_sameby = ["Metadata_Plate"]          # Must match for negative pairs
neg_diffby = ["Metadata_negcon"]         # Must differ for negative pairs

matcher = Matcher(metadata, columns)
pos_pairs = matcher.get_all_pairs(sameby=pos_sameby, diffby=pos_diffby)
neg_pairs = matcher.get_all_pairs(sameby=neg_sameby, diffby=neg_diffby)
```

### Step 3: Similarity Computation

```python
from copairs.compute import cosine_indexed

# Compute pairwise cosine similarities for all pairs
similarities = cosine_indexed(features, pair_indices, batch_size=20000)

# For target matching, use absolute similarity
if anti_match:
    similarities = np.abs(similarities)
```

### Step 4: Rank List Construction

```python
from copairs.map import build_rank_lists

# Build ranked lists per query
# Each query gets: [relevance_flags, ranks] for all compared samples
rel_k_list = build_rank_lists(pos_pairs, neg_pairs)
```

### Step 5: AP Computation with Null Distribution

```python
import copairs.compute_np as backend

# Compute AP scores
ap_scores = rel_k_list.apply(backend.compute_ap)

# Compute null distribution via permutation
null_dists = backend.compute_null_dists(rel_k_list, null_size=100000)

# Compute p-values
p_values = backend.compute_p_values(null_dists, ap_scores, null_size)
```

### Step 6: Aggregation

```python
from copairs.map import aggregate

# Aggregate per-sample AP to per-perturbation mAP
# Also computes q-values (FDR correction)
results = aggregate(per_sample_results, group_by=["Metadata_broad_sample"], threshold=0.05)
```

## Consensus Profiles

For matching tasks, replicate profiles are aggregated into consensus profiles:

```python
def consensus(profiles, group_by):
    """Compute median consensus per perturbation."""
    return profiles.groupby(group_by)[feature_cols].median()
```

This reduces noise and provides a single representative embedding per perturbation.

## Output Files

| File                                 | Description                    |
| ------------------------------------ | ------------------------------ |
| `cellprofiler_replicability_map.csv` | Per-perturbation mAP values    |
| `cellprofiler_replicability_fr.csv`  | Fraction retrieved summary     |
| `cellprofiler_matching_map.csv`      | Target matching mAP values     |
| `cellprofiler_matching_fr.csv`       | Target matching FR summary     |
| `figures/replicability_barplot.png`  | Visualization by modality/cell |

## Algorithm Complexity

| Operation              | Complexity                                |
| ---------------------- | ----------------------------------------- |
| Pair generation        | O(n²) worst case, optimized with indexing |
| Similarity computation | O(p × d) where p = pairs, d = dimensions  |
| Rank list construction | O(n log n) per query                      |
| Permutation test       | O(null_size × n)                          |

For typical datasets (~10k samples, ~1k features), the pipeline runs in minutes.

## References

- [copairs](https://github.com/cytomining/copairs) - Pairwise comparison library
- Chandrasekaran et al. (2024) - CPJUMP1 benchmark methodology
