#!/usr/bin/env bash
set -euo pipefail

# Download MolmoWeb model weights from HuggingFace.
#
# Usage:
#   bash scripts/download_weights.sh                          # downloads MolmoWeb-8B
#   bash scripts/download_weights.sh allenai/MolmoWeb-4B      # downloads MolmoWeb-4B

export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_TELEMETRY=1
export TQDM_DISABLE=1

MODEL="${1:-allenai/MolmoWeb-8B}"
LOCAL_DIR="./checkpoints/$(basename "$MODEL")"

echo "Downloading $MODEL -> $LOCAL_DIR"
uv run hf download "$MODEL" --local-dir "$LOCAL_DIR" --quiet

echo ""
echo "Done. To use this checkpoint, run:"
echo "  export CKPT=\"$LOCAL_DIR\""
