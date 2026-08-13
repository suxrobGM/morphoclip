# CPJUMP1 reference metadata

Small first-party copies of the CPJUMP1 metadata the benchmark harness, split
strategies, and CWA plate-condition map need. Vendored so a fresh clone works
without a manual download step.

| File | Provenance |
| --- | --- |
| `cpjump1_metadata.csv` | Per-well benchmark-eligible metadata (40 plates), from the [Chandrasekaran 2024 CPJUMP1 repo](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1) |
| `experiment-metadata.tsv` | Per-plate experiment design (all 75 plates: cell type, modality, timepoint, density, antibiotics, cell line, time delay), produced by the upstream `0.create-experiment-metadata.ipynb` in the same repo |
| `JUMP-Target-1_compound_metadata_additional_annotations.tsv` | Compound target annotations, from the same repo |

A regenerated `experiment-metadata.tsv` placed in `output/benchmark/input/`
takes precedence over the vendored copy (see
`morphoclip.splits.contexts.resolve_metadata_path`).

The per-plate platemaps and JUMP-Target metadata live in `data/metadata/`,
committed as well; `morphoclip data fetch --metadata` refreshes them from the
public Cell Painting Gallery S3 bucket.
