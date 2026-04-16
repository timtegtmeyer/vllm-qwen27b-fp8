#!/bin/bash
set -e

MODEL_ID="Qwen/Qwen3.5-27B-FP8"
MODEL_DIR="/models/Qwen3.5-27B-FP8"

HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo '')"

if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

echo "Downloading $MODEL_ID to $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

echo "Download complete."
