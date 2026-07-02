"""Console rendering for the CellCLIP training command."""

from pathlib import Path
from typing import Any

from cellclip.training.config import CellCLIPTrainingConfig


def render_train_config(cfg: CellCLIPTrainingConfig, *, config_path: Path, run_dir: Path) -> None:
    """Print the resolved training configuration banner."""
    print("=" * 60)
    print("Local CellCLIP Training")
    print("=" * 60)
    print(f"Config:            {config_path}")
    print(f"Run directory:     {run_dir}")
    print(f"Feature root:      {cfg.dataset.feature_root}")
    print(f"Split strategy:    {cfg.dataset.split_strategy}")
    print(f"Train subset:      {cfg.dataset.subset}")
    print(f"Eval subset:       {cfg.dataset.eval_subset}")
    print(f"Unique perts:      {cfg.dataset.unique_perturbations}")
    print(f"Train max sites:   {cfg.dataset.train_max_sites_per_well}")
    print(f"Eval max sites:    {cfg.dataset.eval_max_sites_per_well}")
    print(f"Within-well interp:{cfg.dataset.within_well_interp_sites}")
    print(f"Same-pert interp:  {cfg.dataset.same_pert_interp_sites}")
    print(f"Interp alpha:      {cfg.dataset.interp_alpha}")
    print(f"Model variant:     {cfg.model.variant}")
    print(f"Text model:        {cfg.model.text_model_name}")
    print(f"Tokenizer:         {cfg.model.tokenizer_name}")
    if cfg.model.variant in {"chemberta_film", "chemberta"}:
        print(f"ChemBERTa model:   {cfg.model.chemberta_model_name}")
        print(f"SMILES tokenizer:  {cfg.model.chemberta_tokenizer_name}")
        print(f"Chem fusion:       {cfg.model.chem_fusion_type}")
        print(f"Prompt policy:     {cfg.model.chem_prompt_policy}")
        print(f"Chem pooling:      {cfg.model.chemberta_pooling}")
        print(f"Freeze ChemBERTa:  {cfg.model.freeze_chemberta}")
        print(f"Tune top layers:   {cfg.model.chemberta_tune_layers}")
    print(f"Loss:              {cfg.optimization.loss_type}")
    print("=" * 60)


def render_train_summary(result: dict[str, Any]) -> None:
    """Print the post-training summary banner."""
    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Train wells:       {result['train_wells']}")
    print(f"Eval wells:        {result['eval_wells']}")
    print(f"Metrics:           {result['metrics_path']}")
    print(f"Best checkpoint:   {result['best_checkpoint']}")
    print(f"Last checkpoint:   {result['last_checkpoint']}")
    print("=" * 60)
