# Report

`main.pdf` is the arXiv preprint. It is tracked so the repo carries the built
paper alongside the sources. Rebuild it with:

```bash
cd report && latexmk -pdf main.tex
```

`latexmk` writes `.aux`, `.log`, `.fls` and friends alongside the PDF. Those are
gitignored; only `main.tex`, `sections/`, `references.bib`, `figures/` and the
built `main.pdf` are tracked. For an arXiv upload, submit `main.tex`,
`sections/`, `references.bib` (or the generated `main.bbl`) and `figures/`, and
leave the build artifacts out.

Retrieval numbers trace to `output/morphoclip_runs/<run>/eval_{val,test}.json`;
standard benchmark numbers trace to `output/benchmark_<run>/`. The cached-feature
figures come from `scripts/features/diagnose_features.py`, and their sampled-plate
metrics are preserved as `figures/feature_metrics.json`. `REVIEW.md` records the
review passes and which findings each rewrite closed.

One caveat on the numbers: the benchmark harness is not reproducible run to run.
Only replicability `mean_average_precision` is deterministic. Fraction-retrieved
varies between two runs of identical code because the matching population is
filtered by a random permutation null, so the reported values carry unquantified
run-to-run variance.
