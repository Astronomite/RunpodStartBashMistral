#!/bin/bash

# Fail fast
set -e

echo "[INFO] Updating system and installing dependencies..."
apt-get update && apt-get install -y curl

echo "[INFO] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Set Ollama model path to persistent volume
export OLLAMA_MODEL_PATH="/workspace/ollama_models"
mkdir -p "$OLLAMA_MODEL_PATH"

echo "[INFO] Starting Ollama server..."
ollama serve --model-path "$OLLAMA_MODEL_PATH" &

# Wait a bit for Ollama to spin up
sleep 10

echo "[INFO] Pulling Mistral (Large 2.1)..."
ollama pull mistral

echo "[INFO] Ready. Container is now idling."
tail -f /dev/null
