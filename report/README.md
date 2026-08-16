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

Every number in the paper traces to a file under `output/` (gitignored, on the
training machine): `output/morphoclip_runs/<run>/eval_{val,test}.json` for
retrieval and `output/benchmark_<run>/` for the standard CPJUMP1 benchmark.
`REVIEW.md` records the review passes and which findings each rewrite closed.
Figures in `figures/` come from `scripts/features/diagnose_features.py`.

One caveat on the numbers: the benchmark harness is not reproducible run to run.
Only replicability `mean_average_precision` is deterministic. Fraction-retrieved
varies between two runs of identical code because the matching population is
filtered by a random permutation null, so the reported values carry unquantified
run-to-run variance.
