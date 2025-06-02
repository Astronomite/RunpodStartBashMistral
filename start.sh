#!/bin/bash

# Fail fast
set -e

echo "[INFO] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

ollama serve &

# Wait a bit for Ollama to spin up
sleep 10

echo "[INFO] Pulling Mistral (Large 2.1)..."
ollama pull mistral-large

pip install requests

echo "[INFO] Ready. Container is now idling."
tail -f /dev/null
