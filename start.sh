#!/bin/bash
set -e
echo "[INFO] Starting full setup for Mistral-Large-2411 Q6_K with llama.cpp..."

# 1. RCLONE SETUP
mkdir -p /workspace/rclone_config
if [ ! -f /workspace/rclone_config/rclone.conf ]; then
    echo "[WARN] No rclone config found. Skipping rclone copy..."
else
    export RCLONE_CONFIG=/workspace/rclone_config/rclone.conf
fi

# 2. SYSTEM + PYTHON DEPS
apt-get update && apt-get install -y git curl wget build-essential cmake software-properties-common
pip install --upgrade pip
pip install huggingface_hub requests

# 3. AUTHENTICATE HUGGINGFACE
# Check for token from RunPod secrets, environment, or stored credentials
if [ -n "$RUNPOD_SECRET_HUGGINGFACE_TOKEN" ]; then
    export HUGGINGFACE_TOKEN="$RUNPOD_SECRET_HUGGINGFACE_TOKEN"
    echo "[INFO] Using HUGGINGFACE_TOKEN from RunPod secrets"
elif [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "[INFO] Using HUGGINGFACE_TOKEN from environment"
elif huggingface-cli whoami > /dev/null 2>&1; then
    echo "[INFO] Using stored Hugging Face credentials"
    unset HUGGINGFACE_TOKEN  # Don't need token if already logged in
else
    echo "[INPUT] Enter your Hugging Face token:"
    echo "[HINT] For security, create a RunPod secret named 'HUGGINGFACE_TOKEN'"
    read -s HUGGINGFACE_TOKEN
    export HUGGINGFACE_TOKEN
    echo "[INFO] Token set from user input"
fi

if [ -z "$HUGGINGFACE_TOKEN" ] && ! huggingface-cli whoami > /dev/null 2>&1; then
    echo "[ERROR] No Hugging Face authentication available. Exiting..."
    exit 1
fi

# 4. CREATE MODEL DIRECTORY
mkdir -p /workspace/models/mistral-large-2411

# 5. DOWNLOAD MISTRAL-LARGE-2411 Q6_K QUANTIZED MODEL
echo "[INFO] Downloading Mistral-Large-2411 Q6_K quantized model..."
python3 - <<EOF
from huggingface_hub import hf_hub_download
import os

# Download the Q6_K quantized model
token = os.environ.get('HUGGINGFACE_TOKEN', None)
model_file = hf_hub_download(
    repo_id="bartowski/Mistral-Large-Instruct-2411-GGUF",
    filename="Mistral-Large-Instruct-2411-Q6_K.gguf",
    local_dir="/workspace/models/mistral-large-2411",
    token=token  # Will be None if using stored credentials
)

print(f"Model downloaded to: {model_file}")
EOF

# 6. INSTALL LLAMA.CPP
echo "[INFO] Installing llama.cpp for GGUF model serving..."
cd /workspace
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support (adjust based on your hardware)
make LLAMA_CUDA=1 -j$(nproc)

# Alternative build commands for different hardware:
# For CPU only: make -j$(nproc)
# For OpenCL: make LLAMA_CLBLAST=1 -j$(nproc)
# For Metal (macOS): make LLAMA_METAL=1 -j$(nproc)

# 7. START LLAMA.CPP SERVER
echo "[INFO] Starting llama.cpp server with Mistral Large 2411 Q6_K..."
cd /workspace/llama.cpp
./server \
    --model /workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K.gguf \
    --host 0.0.0.0 \
    --port 8000 \
    --ctx-size 32768 \
    --threads $(nproc) \
    --chat-template mistral \
    --verbose &

# Wait a moment for server to start
sleep 10

# 8. OPTIONAL RCLONE COPY
if [ -f /workspace/rclone_config/rclone.conf ]; then
    echo "[INFO] Copying RheannaGiftFiles from Google Drive..."
    rclone copy gdrive:RheannaGiftFiles /workspace/ --config /workspace/rclone_config/rclone.conf
fi

echo "[✅] Setup complete. llama.cpp server available at http://localhost:8000"
echo "[INFO] You can now query the model using HTTP requests to the server"
echo "[INFO] Server logs will appear below..."

# Keep the script running and show server logs
tail -f /dev/null
