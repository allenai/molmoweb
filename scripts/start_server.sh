#!/usr/bin/env bash
set -euo pipefail

# Start the MolmoWeb FastAPI model server.
#
# Usage:
#   bash scripts/start_server.sh                                  # defaults: MolmoWeb-8B, hf backend, port 8001
#   bash scripts/start_server.sh allenai/MolmoWeb-4B              # specify model
#   bash scripts/start_server.sh ./checkpoints/MolmoWeb-8B 8002   # local path + custom port
#
# Environment variables (all optional, CLI args take precedence):
#   CKPT             Model checkpoint path or HuggingFace ID
#   PREDICTOR_TYPE   Backend: "hf", "vllm", or "native"
#   PORT             Server port
#   TEMPERATURE      Sampling temperature
#   TOP_P            Top-p sampling

export CKPT="${1:-${CKPT:-allenai/MolmoWeb-8B}}"
PORT="${2:-${PORT:-8001}}"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONWARNINGS="ignore"
export PREDICTOR_TYPE="${PREDICTOR_TYPE:-native}"
export NUM_PREDICTORS="${NUM_PREDICTORS:-1}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.8}"

echo "Starting MolmoWeb server"
echo "  Checkpoint:     $CKPT"
echo "  Backend:        $PREDICTOR_TYPE"
echo "  GPU workers:    $NUM_PREDICTORS"
echo "  Temperature:    $TEMPERATURE"
echo "  Top-p:          $TOP_P"
echo "  Port:           $PORT"
echo ""
echo "Endpoint will be: http://127.0.0.1:$PORT/predict"
echo ""

uv run uvicorn agent.fastapi_model_server:app --host 0.0.0.0 --port "$PORT"
