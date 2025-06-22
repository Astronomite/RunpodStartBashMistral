#!/bin/bash
set -e
echo "[INFO] Starting full setup for Mistral-Large-2411 Q6_K with llama.cpp..."

# Check system capabilities
echo "[INFO] Checking system capabilities..."
if nvidia-smi > /dev/null 2>&1; then
    echo "[INFO] ✅ NVIDIA GPU detected"
    
    # Count available GPUs
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    echo "[INFO] Found $GPU_COUNT GPU(s)"
    
    # Show GPU details
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    
    GPU_AVAILABLE=true
    
    # Calculate tensor splits for available GPUs
    if [ "$GPU_COUNT" -eq 1 ]; then
        TENSOR_SPLIT=""
        PARALLEL_REQUESTS=4
    elif [ "$GPU_COUNT" -eq 2 ]; then
        TENSOR_SPLIT="--tensor-split 0.5,0.5"
        PARALLEL_REQUESTS=8
    elif [ "$GPU_COUNT" -eq 3 ]; then
        TENSOR_SPLIT="--tensor-split 0.33,0.33,0.34"
        PARALLEL_REQUESTS=12
    elif [ "$GPU_COUNT" -eq 4 ]; then
        TENSOR_SPLIT="--tensor-split 0.25,0.25,0.25,0.25"
        PARALLEL_REQUESTS=16
    else
        # For more than 4 GPUs, calculate even splits
        SPLIT_VALUE=$(python3 -c "print(f'{1.0/$GPU_COUNT:.3f}' * ($GPU_COUNT-1) + f',{1.0/$GPU_COUNT:.3f}')")
        TENSOR_SPLIT="--tensor-split $SPLIT_VALUE"
        PARALLEL_REQUESTS=$((GPU_COUNT * 4))
    fi
    
    echo "[INFO] Tensor split configured for $GPU_COUNT GPUs: $TENSOR_SPLIT"
    echo "[INFO] Parallel requests: $PARALLEL_REQUESTS"
    
else
    echo "[WARN] ⚠️  No NVIDIA GPU detected - will use CPU mode"
    GPU_AVAILABLE=false
    GPU_COUNT=0
    PARALLEL_REQUESTS=2  # Conservative for CPU
fi

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

# First, try to download using huggingface-cli (more reliable for large files)
if command -v huggingface-cli &> /dev/null; then
    echo "[INFO] Using huggingface-cli to download the model..."
    
    # Set the token if available
    if [ -n "$HUGGINGFACE_TOKEN" ]; then
        huggingface-cli login --token "$HUGGINGFACE_TOKEN"
    fi
    
    # Download the Q6_K model (it's split into multiple files)
    echo "[INFO] Downloading Q6_K quantized model files..."
    huggingface-cli download bartowski/Mistral-Large-Instruct-2411-GGUF \
        --include "Mistral-Large-Instruct-2411-Q6_K/*" \
        --local-dir /workspace/models/mistral-large-2411 \
        --local-dir-use-symlinks False
    
    # Check if download was successful
    if [ -d "/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K" ]; then
        echo "[INFO] ✅ Q6_K model downloaded successfully"
        MODEL_PATH="/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K"
    else
        echo "[WARN] Q6_K download failed, trying Q4_K_M as fallback..."
        huggingface-cli download bartowski/Mistral-Large-Instruct-2411-GGUF \
            --include "Mistral-Large-Instruct-2411-Q4_K_M/*" \
            --local-dir /workspace/models/mistral-large-2411 \
            --local-dir-use-symlinks False
        MODEL_PATH="/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q4_K_M"
    fi
else
    # Fallback to Python download
    echo "[INFO] Using Python to download the model..."
    python3 - <<EOF
from huggingface_hub import snapshot_download
import os

# Download the Q6_K quantized model
token = os.environ.get('HUGGINGFACE_TOKEN', None)

try:
    model_path = snapshot_download(
        repo_id="bartowski/Mistral-Large-Instruct-2411-GGUF",
        allow_patterns=["Mistral-Large-Instruct-2411-Q6_K/*"],
        local_dir="/workspace/models/mistral-large-2411",
        token=token,
        local_dir_use_symlinks=False
    )
    print(f"Q6_K model downloaded to: {model_path}")
except Exception as e:
    print(f"Q6_K download failed: {e}")
    print("Trying Q4_K_M as fallback...")
    model_path = snapshot_download(
        repo_id="bartowski/Mistral-Large-Instruct-2411-GGUF",
        allow_patterns=["Mistral-Large-Instruct-2411-Q4_K_M/*"],
        local_dir="/workspace/models/mistral-large-2411",
        token=token,
        local_dir_use_symlinks=False
    )
    print(f"Q4_K_M model downloaded to: {model_path}")
EOF
    
    # Determine which model was downloaded
    if [ -d "/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K" ]; then
        MODEL_PATH="/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K"
    else
        MODEL_PATH="/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q4_K_M"
    fi
fi

# 6. INSTALL LLAMA.CPP
echo "[INFO] Installing llama.cpp for GGUF model serving..."
cd /workspace
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Check if NVIDIA GPU is available and build accordingly
if [ "$GPU_AVAILABLE" = true ]; then
    echo "[INFO] Building llama.cpp with CUDA support..."
    make LLAMA_CUDA=1 -j$(nproc)
else
    echo "[INFO] Building llama.cpp for CPU only..."
    make -j$(nproc)
fi

# 7. START LLAMA.CPP SERVER WITH PARALLELISM
echo "[INFO] Starting llama.cpp server with Mistral Large 2411..."
cd /workspace/llama.cpp

# Calculate optimal thread counts
TOTAL_THREADS=$(nproc)
GPU_LAYERS=99  # Use all GPU layers if GPU available

# Find the actual model file (it might be split)
if [ -f "$MODEL_PATH"/*.gguf ]; then
    # Single file model
    MODEL_FILE=$(find "$MODEL_PATH" -name "*.gguf" -type f | head -1)
    echo "[INFO] Using single model file: $MODEL_FILE"
else
    # Check if it's the original model directory structure
    if [ -f "/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K.gguf" ]; then
        MODEL_FILE="/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q6_K.gguf"
    elif [ -f "/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q4_K_M.gguf" ]; then
        MODEL_FILE="/workspace/models/mistral-large-2411/Mistral-Large-Instruct-2411-Q4_K_M.gguf"
    else
        # Look for any .gguf file
        MODEL_FILE=$(find /workspace/models/mistral-large-2411 -name "*.gguf" -type f | head -1)
    fi
    echo "[INFO] Using model file: $MODEL_FILE"
fi

if [ -z "$MODEL_FILE" ] || [ ! -f "$MODEL_FILE" ]; then
    echo "[ERROR] No model file found! Check the download."
    exit 1
fi

if [ "$GPU_AVAILABLE" = true ]; then
    echo "[INFO] Starting with GPU acceleration and auto-configured parallelism..."
    ./server \
        --model "$MODEL_FILE" \
        --host 0.0.0.0 \
        --port 8000 \
        --ctx-size 32768 \
        --threads $TOTAL_THREADS \
        --threads-batch $TOTAL_THREADS \
        --gpu-layers $GPU_LAYERS \
        $TENSOR_SPLIT \
        --main-gpu 0 \
        --split-mode layer \
        --chat-template mistral \
        --parallel $PARALLEL_REQUESTS \
        --cont-batching \
        --verbose &
else
    echo "[INFO] Starting with CPU parallelism only..."
    ./server \
        --model "$MODEL_FILE" \
        --host 0.0.0.0 \
        --port 8000 \
        --ctx-size 32768 \
        --threads $TOTAL_THREADS \
        --threads-batch $TOTAL_THREADS \
        --chat-template mistral \
        --parallel $PARALLEL_REQUESTS \
        --cont-batching \
        --verbose &
fi

# Wait a moment for server to start
sleep 10

# 8. OPTIONAL RCLONE COPY
if [ -f /workspace/rclone_config/rclone.conf ]; then
    echo "[INFO] Copying RheannaGiftFiles from Google Drive..."
    rclone copy gdrive:RheannaGiftFiles /workspace/ --config /workspace/rclone_config/rclone.conf
fi

echo "[✅] Setup complete. llama.cpp server available at http://localhost:8000"
echo "[INFO] Model loaded: $MODEL_FILE"
echo "[INFO] You can now query the model using HTTP requests to the server"
echo "[INFO] Server logs will appear below..."

# Keep the script running and show server logs
tail -f /dev/null
