---
paths:
  - "configs/**"
  - "src/morphoclip/config.py"
  - "src/morphoclip/training/config.py"
  - "src/morphoclip/data/config.py"
  - "src/cellclip/training/config.py"
  - "tests/config/**"
---

# Configs

- Subclass `StrictModel` for every config model.
- Load configs only through `morphoclip.config.load_config`.
- Resolve `extends` in the loader, never in a pydantic validator. Configs read
  back from checkpoints must not re-resolve `extends`.
- Express a variant as a `--set` override documented in a comment or README,
  not as a new file that differs by a few keys.
- Keep feature dimensions and untuned constants (`EMBED_DIM`, AdamW betas,
  grad clip, `LOGIT_SCALE_MAX`) as module constants in `training/config.py`,
  not config fields.
- Every committed config needs a golden in `tests/config/goldens/`. After an
  intended resolution change, or adding or removing a config, regenerate and
  review the diff:
  `MORPHOCLIP_REGEN_GOLDENS=1 uv run pytest tests/config/test_golden_resolution.py`
- Treat an unexpected golden change as a bug.
