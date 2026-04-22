#!/bin/bash
set -e

DATA="data/playroom"
OUTPUT="outputs/playroom_v2"

cd "$(dirname "$0")"

python3 train.py \
    --data "$DATA" \
    --output "$OUTPUT" \
    --device cuda \
    --seed 42 \
    --image-scale 1 \
    --iterations 30000 \
    --max-points 37005 \
    --max-gaussians 1000000 \
    --holdout-every 8 \
    --sh-degree 3 \
    --sh-upgrade-every 1000 \
    --l1-weight 0.8 \
    --dssim-weight 0.2 \
    --opacity-reg 0.0 \
    --scale-reg 0.0 \
    --position-lr-init 1.6e-4 \
    --position-lr-final 1.6e-6 \
    --position-lr-delay-mult 0.01 \
    --position-lr-max-steps 30000 \
    --feature-lr 2.5e-3 \
    --opacity-lr 2.5e-2 \
    --scaling-lr 5e-3 \
    --rotation-lr 1e-3 \
    --densify-from 500 \
    --densify-until 15000 \
    --densify-interval 100 \
    --densify-grad-threshold 2e-4 \
    --densify-scale-threshold 0.01 \
    --prune-opacity-threshold 0.005 \
    --prune-screen-radius 20.0 \
    --world-prune-scale 0.1 \
    --split-factor 2 \
    --split-scale-factor 1.6 \
    --clone-jitter 0.0 \
    --opacity-reset-interval 3000 \
    --opacity-reset-max 0.01 \
    --log-every 100 \
    --preview-every 500 \
    --save-every 5000 \
    --preview-index 0

echo "Training done. Checkpoint: $OUTPUT/checkpoints/latest.pt"
echo "Preview images: $OUTPUT/previews/"
