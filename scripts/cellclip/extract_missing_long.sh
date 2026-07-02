#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

batch_size="${1:-16}"

plates=(
  BR00117000
  BR00117003
  BR00117004
  BR00117005
  BR00117006
  BR00117010
  BR00117011
  BR00117012
  BR00117013
  BR00117015
  BR00117016
  BR00117017
  BR00117019
  BR00118039
  BR00118040
  BR00118050
)

for plate in "${plates[@]}"; do
  uv run python scripts/features/extract_features.py \
    --config configs/dataset.yml \
    --plate "$plate" \
    --compressed-root data/raw_compressed \
    --features-root data/features_cellclip_base \
    --model-name facebook/dinov2-giant \
    --device cuda \
    --batch-size "$batch_size" \
    --no-tensors
done
