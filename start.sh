#!/bin/bash

# Fail fast
set -e

# Create rclone config directory if needed
mkdir -p /workspace/rclone_config

# If config doesn't exist, prompt user to create or restore
if [ ! -f /workspace/rclone_config/rclone.conf ]; then
    echo "[WARN] No rclone config found at /workspace/rclone_config/rclone.conf"
    echo "Run 'rclone config --config /workspace/rclone_config/rclone.conf' to set it up."
fi

# Point rclone to the persistent config
export RCLONE_CONFIG=/workspace/rclone_config/rclone.conf

# Define Ollama model storage directory and binary cache
export OLLAMA_MODELS=/workspace/ollama
export PATH=/workspace/ollama_bin:$PATH
mkdir -p "$OLLAMA_MODELS" /workspace/ollama_bin

MODEL_NAME="mistral-large"
MODEL_VARIANT="123b"

# Check available disk space (~60GB needed for mistral-large:123b)
required_space=60000000  # ~60GB in KB
available_space=$(df -k /workspace | tail -1 | awk '{print $4}')
if [ "$available_space" -lt "$required_space" ]; then
    echo "[ERROR] Insufficient disk space in /workspace. Need ~60GB, available: $((available_space/1024))MB."
    exit 1
fi

# Check for Ollama in local persistent bin
echo "[INFO] Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "[INFO] Ollama not found in PATH. Installing locally to /workspace/ollama_bin..."
    curl -fsSL https://ollama.com/install.sh | OLLAMA_DIR=/workspace/ollama_bin sh || {
        echo "[ERROR] Failed to install Ollama."
        exit 1
    }
    if ! command -v ollama &> /dev/null; then
        echo "[ERROR] Ollama installation failed. Binary still not found."
        exit 1
    fi
else
    echo "[INFO] Ollama is already available."
fi

# Start Ollama server
echo "[INFO] Starting Ollama server..."
ollama serve &

# Wait for Ollama server to be ready
echo "[INFO] Waiting for Ollama server to be ready..."
timeout=60
elapsed=0
until curl -s http://localhost:11434 >/dev/null; do
    if [ $elapsed -ge $timeout ]; then
        echo "[ERROR] Ollama server did not start within $timeout seconds."
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done
echo "[INFO] Ollama server is ready."

# Check if Mistral Large 123B is already downloaded
if ollama list | grep -q "$MODEL_NAME:$MODEL_VARIANT"; then
    echo "[INFO] Mistral Large 123B already exists. Skipping download."
else
    echo "[INFO] Pulling Mistral Large 123B..."
    ollama pull "$MODEL_NAME:$MODEL_VARIANT"
fi

# Ensure Python dependencies
pip3 install requests

# Conditionally run rclone copy if config exists
if [ -f /workspace/rclone_config/rclone.conf ]; then
    echo "[INFO] Copying RheannaGiftFiles from Google Drive..."
    rclone copy gdrive:RheannaGiftFiles /workspace/ --config /workspace/rclone_config/rclone.conf
else
    echo "[WARN] Skipping rclone copy. Config file not found."
fi

echo "[INFO] Setup complete. Container is now idling."
tail -f /dev/null
