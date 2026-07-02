#!/usr/bin/env python
"""Train a local CellCLIP model on a CPJUMP1 split subset."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellclip.training import load_training_config, train_cellclip  # noqa: E402

DEFAULT_CONFIG_PATH = Path("configs/cellclip/cellclip_jumpcp.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable DDP multi-GPU training (requires torchrun launcher)",
    )
    return parser


def _resolve_run_dir(config, *, run_name: str | None, output_dir: Path | None) -> Path:
    base_output = output_dir or (PROJECT_ROOT / config.runtime.output_root)
    resolved_name = run_name or config.runtime.run_name
    if resolved_name is None:
        resolved_name = datetime.now().strftime("cellclip_%Y%m%d_%H%M%S")
    return base_output / resolved_name


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_training_config(args.config)
    if args.distributed:
        config.distributed.enabled = True
    if args.output_dir is not None:
        config.runtime.output_root = str(args.output_dir)
    if args.run_name is not None:
        config.runtime.run_name = args.run_name
    if args.split_manifest is not None:
        config.dataset.split_manifest_path = str(args.split_manifest)

    run_dir = _resolve_run_dir(config, run_name=args.run_name, output_dir=args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)

    print("=" * 60)
    print("Local CellCLIP Training")
    print("=" * 60)
    print(f"Config:            {args.config}")
    print(f"Run directory:     {run_dir}")
    print(f"Feature root:      {config.dataset.feature_root}")
    print(f"Split strategy:    {config.dataset.split_strategy}")
    print(f"Train subset:      {config.dataset.subset}")
    print(f"Eval subset:       {config.dataset.eval_subset}")
    print(f"Unique perts:      {config.dataset.unique_perturbations}")
    print(f"Train max sites:   {config.dataset.train_max_sites_per_well}")
    print(f"Eval max sites:    {config.dataset.eval_max_sites_per_well}")
    print(f"Within-well interp:{config.dataset.within_well_interp_sites}")
    print(f"Same-pert interp:  {config.dataset.same_pert_interp_sites}")
    print(f"Interp alpha:      {config.dataset.interp_alpha}")
    print(f"Model variant:     {config.model.variant}")
    print(f"Text model:        {config.model.text_model_name}")
    print(f"Tokenizer:         {config.model.tokenizer_name}")
    if config.model.variant in {"chemberta_film", "chemberta"}:
        print(f"ChemBERTa model:   {config.model.chemberta_model_name}")
        print(f"SMILES tokenizer:  {config.model.chemberta_tokenizer_name}")
        print(f"Chem fusion:       {config.model.chem_fusion_type}")
        print(f"Prompt policy:     {config.model.chem_prompt_policy}")
        print(f"Chem pooling:      {config.model.chemberta_pooling}")
        print(f"Freeze ChemBERTa:  {config.model.freeze_chemberta}")
        print(f"Tune top layers:   {config.model.chemberta_tune_layers}")
    print(f"Loss:              {config.optimization.loss_type}")
    print("=" * 60)

    result = train_cellclip(config, run_dir=run_dir)

    print("=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Train wells:       {result['train_wells']}")
    print(f"Eval wells:        {result['eval_wells']}")
    print(f"Metrics:           {result['metrics_path']}")
    print(f"Best checkpoint:   {result['best_checkpoint']}")
    print(f"Last checkpoint:   {result['last_checkpoint']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
