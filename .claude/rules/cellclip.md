---
paths:
  - "src/cellclip/**"
  - "configs/cellclip/**"
  - "src/morphoclip/cli/cellclip_cmds.py"
---

# CellCLIP

- Keep CellCLIP runnable. Do not extend it, refactor it or add tests for it.
- Never touch `QuickGELU`, `ResidualAttentionBlock` or `LayerNorm` in
  `cellclip/model.py`. The published checkpoint requires an exact state-dict match.
- Do not move CellCLIP code into `morphoclip.models`.
- When changing `morphoclip.training.{distributed,metrics,optim,tb_logger}`,
  keep CellCLIP training and resuming. Keep `split_params(decay_first=False)`
  ordering intact for its old checkpoints.
- Add historical config variants as `--set` commands in
  `configs/cellclip/README.md`, not as new files.
