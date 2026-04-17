#!/bin/bash
set -e

MODEL_ID="Qwen/Qwen3.5-27B-FP8"
MODEL_DIR="/models/Qwen3.5-27B-FP8"

HF_TOKEN="$(cat /run/secrets/HF_TOKEN 2>/dev/null || echo '')"

if [ -n "$HF_TOKEN" ]; then
    if command -v hf &>/dev/null; then
        hf auth login --token "$HF_TOKEN"
    else
        huggingface-cli login --token "$HF_TOKEN"
    fi
fi

echo "Downloading $MODEL_ID to $MODEL_DIR ..."
mkdir -p "$MODEL_DIR"

if command -v hf &>/dev/null; then
    HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_ID" \
        --local-dir "$MODEL_DIR"
else
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

echo "Download complete."
