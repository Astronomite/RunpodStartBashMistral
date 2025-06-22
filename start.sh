#!/bin/bash
set -e

# Configuration
MODEL_DIR="/workspace/models/mistral-large-2411"
LLAMA_CPP_DIR="/workspace/llama.cpp"
SETUP_COMPLETE_FLAG="/workspace/.mistral_setup_complete"

echo "[INFO] Starting Mistral-Large-2411 Q6_K setup..."

# Check if this is a fresh setup or restart
if [ -f "$SETUP_COMPLETE_FLAG" ]; then
    echo "[INFO] 🚀 Setup already complete - Fast startup mode!"
    FAST_STARTUP=true
else
    echo "[INFO] 📦 First time setup - Download mode!"
    FAST_STARTUP=false
fi

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
        SPLIT_VALUE=$(python3 -c "print(','.join([f'{1.0/$GPU_COUNT:.3f}'] * $GPU_COUNT))")
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

# FAST STARTUP PATH - Skip downloads if everything exists
if [ "$FAST_STARTUP" = true ]; then
    echo "[INFO] ⚡ Fast startup - verifying existing installation..."
    
    # Quick verification that key components exist
    if [ ! -d "$LLAMA_CPP_DIR" ] || [ ! -f "$LLAMA_CPP_DIR/server" ]; then
        echo "[WARN] llama.cpp missing, rebuilding..."
        FAST_STARTUP=false
    elif [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR"/*.gguf ]; then
        echo "[WARN] Model files missing, re-downloading..."
        FAST_STARTUP=false
    else
        echo "[INFO] ✅ All components verified - proceeding to server startup"
        # Skip to server startup section
        cd "$LLAMA_CPP_DIR"
        
        # Find the model file
        MODEL_FILE=$(find "$MODEL_DIR" -name "*.gguf" -type f | head -1)
        if [ -z "$MODEL_FILE" ]; then
            echo "[ERROR] No model file found!"
            exit 1
        fi
        
        echo "[INFO] 🚀 Starting llama.cpp server immediately..."
        exec ./start_server.sh "$MODEL_FILE" "$GPU_AVAILABLE" "$GPU_COUNT" "$TENSOR_SPLIT" "$PARALLEL_REQUESTS"
    fi
fi

# FULL SETUP PATH - Only run on first setup or if verification failed
echo "[INFO] 🔧 Running full setup..."

# 1. RCLONE SETUP (only if not fast startup)
mkdir -p /workspace/rclone_config
if [ ! -f /workspace/rclone_config/rclone.conf ]; then
    echo "[WARN] No rclone config found. Skipping rclone copy..."
else
    export RCLONE_CONFIG=/workspace/rclone_config/rclone.conf
fi

# 2. SYSTEM + PYTHON DEPS
echo "[INFO] Installing system dependencies..."
apt-get update && apt-get install -y git curl wget build-essential cmake software-properties-common
pip install --upgrade pip
pip install huggingface_hub requests

# 3. AUTHENTICATE HUGGINGFACE
echo "[INFO] Setting up Hugging Face authentication..."
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
mkdir -p "$MODEL_DIR"

# 5. DOWNLOAD MISTRAL-LARGE-2411 Q6_K QUANTIZED MODEL
echo "[INFO] Checking available disk space..."
df -h /workspace

echo "[INFO] Downloading Mistral-Large-2411 Q6_K quantized model (~90GB)..."
echo "[INFO] This will take a while on CPU-only pod - perfect for initial download!"

# Function to download with retry and resume
download_with_retry() {
    local repo_id="$1"
    local local_dir="$2"
    local max_retries=5
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        echo "[INFO] Q6_K download attempt $((retry_count + 1))/$max_retries"
        
        # Use Python with huggingface_hub for most reliable downloads
        python3 - <<EOF
from huggingface_hub import hf_hub_download
import os
import time

token = os.environ.get('HUGGINGFACE_TOKEN', None)
files = [
    "Mistral-Large-Instruct-2411-Q6_K/Mistral-Large-Instruct-2411-Q6_K-00001-of-00003.gguf",
    "Mistral-Large-Instruct-2411-Q6_K/Mistral-Large-Instruct-2411-Q6_K-00002-of-00003.gguf", 
    "Mistral-Large-Instruct-2411-Q6_K/Mistral-Large-Instruct-2411-Q6_K-00003-of-00003.gguf"
]

try:
    for filename in files:
        print(f"[INFO] Downloading {filename}...")
        hf_hub_download(
            repo_id="$repo_id",
            filename=filename,
            local_dir="$local_dir",
            token=token,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[INFO] ✅ {filename} completed")
        time.sleep(2)  # Brief pause between files
    print("[INFO] ✅ All model files downloaded successfully")
except Exception as e:
    print(f"[ERROR] Download failed: {e}")
    exit(1)
EOF
        
        # Check if download succeeded
        if [ $? -eq 0 ]; then
            echo "[INFO] ✅ Q6_K download completed successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "[WARN] Download failed, waiting 60 seconds before retry..."
                sleep 60
            fi
        fi
    done
    
    echo "[ERROR] Q6_K download failed after $max_retries attempts"
    return 1
}

# Download Q6_K model with retry logic
if download_with_retry "bartowski/Mistral-Large-Instruct-2411-GGUF" "$MODEL_DIR"; then
    echo "[INFO] ✅ Model downloaded successfully to $MODEL_DIR"
else
    echo "[ERROR] Failed to download model after all retry attempts!"
    exit 1
fi

# 6. INSTALL LLAMA.CPP
echo "[INFO] Installing llama.cpp for GGUF model serving..."
cd /workspace
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
fi
cd llama.cpp

# Build llama.cpp (will rebuild with GPU support when restarted with GPUs)
if [ "$GPU_AVAILABLE" = true ]; then
    echo "[INFO] Building llama.cpp with CUDA support..."
    make clean
    make LLAMA_CUDA=1 -j$(nproc)
else
    echo "[INFO] Building llama.cpp for CPU (will rebuild with CUDA when GPUs available)..."
    make clean
    make -j$(nproc)
fi

# 7. CREATE SERVER STARTUP SCRIPT
echo "[INFO] Creating optimized server startup script..."
cat > /workspace/llama.cpp/start_server.sh << 'SCRIPT_EOF'
#!/bin/bash
set -e

MODEL_FILE="$1"
GPU_AVAILABLE="$2"
GPU_COUNT="$3"
TENSOR_SPLIT="$4"
PARALLEL_REQUESTS="$5"

echo "[INFO] 🚀 Starting llama.cpp server with optimized settings..."
echo "[INFO] Model: $MODEL_FILE"
echo "[INFO] GPU Available: $GPU_AVAILABLE"
echo "[INFO] GPU Count: $GPU_COUNT"

# Calculate optimal settings
TOTAL_THREADS=$(nproc)
GPU_LAYERS=99  # Use all GPU layers if GPU available

if [ "$GPU_AVAILABLE" = "true" ]; then
    echo "[INFO] 🔥 Starting with GPU acceleration..."
    exec ./server \
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
        --verbose
else
    echo "[INFO] 💻 Starting with CPU only..."
    exec ./server \
        --model "$MODEL_FILE" \
        --host 0.0.0.0 \
        --port 8000 \
        --ctx-size 32768 \
        --threads $TOTAL_THREADS \
        --threads-batch $TOTAL_THREADS \
        --chat-template mistral \
        --parallel $PARALLEL_REQUESTS \
        --cont-batching \
        --verbose
fi
SCRIPT_EOF

chmod +x /workspace/llama.cpp/start_server.sh

# 8. OPTIONAL RCLONE COPY
if [ -f /workspace/rclone_config/rclone.conf ]; then
    echo "[INFO] Copying RheannaGiftFiles from Google Drive..."
    rclone copy gdrive:RheannaGiftFiles /workspace/ --config /workspace/rclone_config/rclone.conf
fi

# 9. MARK SETUP AS COMPLETE
echo "[INFO] ✅ Marking setup as complete..."
touch "$SETUP_COMPLETE_FLAG"

# 10. START SERVER OR PROVIDE INSTRUCTIONS
if [ "$GPU_COUNT" -eq 0 ]; then
    echo ""
    echo "🎉 [SUCCESS] Download and setup complete!"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "   1. Stop this pod"
    echo "   2. Start a new pod with your desired GPU configuration"
    echo "   3. Run this script again - it will start instantly!"
    echo ""
    echo "💾 Everything is saved in /workspace:"
    echo "   - Model: $MODEL_DIR"
    echo "   - llama.cpp: $LLAMA_CPP_DIR" 
    echo "   - Setup flag: $SETUP_COMPLETE_FLAG"
    echo ""
    echo "⚡ On GPU restart, the server will start in <30 seconds!"
    
    # Don't start server on CPU-only setup, just exit cleanly
    exit 0
else
    echo "[INFO] 🚀 GPUs detected - starting server immediately..."
    
    # Find the model file
    MODEL_FILE=$(find "$MODEL_DIR" -name "*.gguf" -type f | head -1)
    if [ -z "$MODEL_FILE" ]; then
        echo "[ERROR] No model file found!"
        exit 1
    fi
    
    echo "[INFO] Using model: $MODEL_FILE"
    cd "$LLAMA_CPP_DIR"
    
    # Rebuild with CUDA if we have GPUs but server wasn't built with CUDA
    if [ ! -f ".cuda_built" ]; then
        echo "[INFO] Rebuilding llama.cpp with CUDA support..."
        make clean
        make LLAMA_CUDA=1 -j$(nproc)
        touch .cuda_built
    fi
    
    echo "[✅] Setup complete. Starting llama.cpp server at http://localhost:8000"
    exec ./start_server.sh "$MODEL_FILE" "$GPU_AVAILABLE" "$GPU_COUNT" "$TENSOR_SPLIT" "$PARALLEL_REQUESTS"
fi
