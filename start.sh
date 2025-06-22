#!/bin/bash

set -e

echo "[INFO] Starting full setup for Mistral-Large-2407 with vLLM..."

# 1. RCLONE SETUP
mkdir -p /workspace/rclone_config
if [ ! -f /workspace/rclone_config/rclone.conf ]; then
    echo "[WARN] No rclone config found. Skipping rclone copy..."
else
    export RCLONE_CONFIG=/workspace/rclone_config/rclone.conf
fi

# 2. SYSTEM + PYTHON DEPS
apt-get update && apt-get install -y git curl wget software-properties-common
pip install --upgrade pip
pip install torch transformers accelerate huggingface_hub vllm

# 3. AUTHENTICATE HUGGINGFACE (Replace with your token or use env var)
export HUGGINGFACE_TOKEN=hf_your_token_here

# 4. DOWNLOAD MISTRAL-LARGE-2407
echo "[INFO] Downloading Mistral-Large-Instruct-2407..."
python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="mistralai/Mistral-Large-Instruct-2407",
  local_dir="/workspace/models/mistral-large-2407",
  token="${HUGGINGFACE_TOKEN}",
  allow_patterns=[
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "params.json",
    "consolidated.*.safetensors"
  ]
)
EOF

# 5. START VLLM SERVER
echo "[INFO] Launching vLLM OpenAI-compatible server on all GPUs..."
python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/mistral-large-2407 \
  --tensor-parallel-size 5 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --port 8000 &

# 6. OPTIONAL RCLONE COPY
if [ -f /workspace/rclone_config/rclone.conf ]; then
    echo "[INFO] Copying RheannaGiftFiles from Google Drive..."
    rclone copy gdrive:RheannaGiftFiles /workspace/ --config /workspace/rclone_config/rclone.conf
fi

echo "[✅] Setup complete. vLLM API available at http://localhost:8000/v1"
tail -f /dev/null
