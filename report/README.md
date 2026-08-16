# Report

The arXiv preprint. Only the sources are tracked: `main.tex`, `sections/`,
`references.bib` and `figures/`. Every build artifact, `main.pdf` included, goes
to `report/build/`, which is gitignored.

```bash
uv run poe report          # build report/build/main.pdf
uv run poe report-arxiv    # build, then pack report/build/arxiv.tar.gz
uv run poe report-clean    # delete report/build/
```

`report/.latexmkrc` sets the output directory and the bibtex search path, so
`cd report && latexmk -pdf main.tex` gives the same result. The poe tasks pass
it with `-r` because latexmk reads the file from the working directory, which it
changes only afterwards.

`report-arxiv` stages `main.tex`, `sections/`, the figures that `\includegraphics`
actually names, and the compiled `main.bbl`. arXiv never runs bibtex, which is
why the tarball carries `main.bbl` instead of `references.bib`.

Retrieval numbers trace to `output/morphoclip_runs/<run>/eval_{val,test}.json`;
standard benchmark numbers trace to `output/benchmark_<run>/`. The cached-feature
figures come from `scripts/features/diagnose_features.py`, and their sampled-plate
metrics are preserved as `figures/feature_metrics.json`. `architecture.pdf` comes
from `figures/make_architecture.py`. `REVIEW.md` records the review passes and
which findings each rewrite closed.

One caveat on the numbers: the benchmark harness is not reproducible run to run.
Only replicability `mean_average_precision` is deterministic. Fraction-retrieved
varies between two runs of identical code because the matching population is
filtered by a random permutation null, so the reported values carry unquantified
run-to-run variance.
