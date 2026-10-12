# MorphoCLIP

CLIP-style model aligning Cell Painting images (frozen DINOv3 features) with
perturbation text (frozen BioClinical ModernBERT) on CPJUMP1. Packages:
`morphoclip` (active), `benchmark` (standalone, frozen), `cellclip` (frozen
baseline).

## Commands

```bash
uv sync --extra cu128 --extra benchmark   # torch extra: cpu | cu128 | cu130
uv run poe check                          # run before finishing any change
uv run poe test
uv run poe test-realdata                  # needs downloaded data

uv run morphoclip --help                  # every pipeline step is a subcommand
uv run morphoclip train --config configs/train/base.yaml
uv run morphoclip text precompute
uv run morphoclip benchmark --config configs/benchmark.yml
uv run torchrun --nproc_per_node=4 -m morphoclip.cli train --config configs/train/ddp.yaml --distributed

uv run poe report                         # paper: report/build/main.pdf
uv run poe report-tmlr                    # TMLR PDF + anonymized code zip (report/README.md)
cd docs && bun run build                  # docs site
```

## Rules

- Change dependencies only through uv (`uv add`, `uv remove`). Never edit
  `uv.lock` by hand.
- Run `uv run poe check` before declaring a change done.
- Do not modify `src/cellclip/` beyond keeping it runnable.
- Do not add torch, morphoclip or cellclip imports to `src/benchmark/`.
- Do not unfreeze BERT or DINOv3.
- Do not change the md5 bucketing in `splits/strategies.py`. It invalidates
  every saved split.
- Do not turn `EMBED_DIM`, `TEXT_INPUT_DIM` or `INPUT_CHANNELS` into config
  fields. They are fixed by the cached features.
- Keep every image aggregator invariant to site order.
- Keep `ProjectionHead` output L2-normalized.
- After editing a template in `models/prompts.py`, rerun
  `morphoclip text precompute`. The cache is keyed by perturbation id, so stale
  embeddings load silently.
- Keep `models/prompts.py` (BERT prompts) and `data/perturbation.py` (display
  labels, `text_level`) separate. `build_prompt_from_info` is the only bridge.
- Do not treat benchmark differences between runs as regressions, and do not
  try to make benchmark output deterministic. Only replicability
  `mean_average_precision` is reproducible.
- Never read `data/` from tests or commit anything under it except
  `data/reference/` and `data/metadata/`.
- Ask before running `data fetch` or `features extract`. They write about
  3 GB per plate across 51 plates.
