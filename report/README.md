# Report

`main.pdf` is the deliverable and is tracked deliberately: there is no release
channel to publish it to, and it is the record of the results the frozen
CellCLIP baseline and the benchmark numbers come from.

Build it with:

```bash
cd report && latexmk -pdf main.tex
```

`latexmk` writes `.aux`, `.log`, `.fls` and friends alongside the PDF. Those are
gitignored; only `main.tex`, `sections/`, `references.bib`, `figures/` and the
built `main.pdf` are tracked.

`REVIEW.md` is the write-up review notes. Figures in `figures/` come from
`scripts/features/diagnose_features.py` and the benchmark plotting code.

One caveat on the numbers: the benchmark harness is not reproducible run to run.
Only replicability `mean_average_precision` is deterministic. Fraction-retrieved
varies between two runs of identical code because the matching population is
filtered by a random permutation null, so the reported values carry unquantified
run-to-run variance.
