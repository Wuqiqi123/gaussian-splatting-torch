#!/bin/bash
set -e

DATA="data/playroom"
OUTPUT="outputs/playroom_v2"
CHECKPOINT="$OUTPUT/checkpoints/best.pt"
RENDER_OUT="$OUTPUT/render_val"

cd "$(dirname "$0")"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi

python3 render.py \
    --checkpoint "$CHECKPOINT" \
    --data "$DATA" \
    --output "$RENDER_OUT" \
    --split val \
    --limit 30 \
    --image-scale 1 \
    --device cuda

echo "Rendered to: $RENDER_OUT"
