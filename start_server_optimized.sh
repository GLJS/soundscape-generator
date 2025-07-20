#!/bin/bash

# Optimized Audio Flamingo HTTP Server Startup Script
# This script starts the optimized FastAPI server with all performance enhancements

# Configuration
export AUDIO_FLAMINGO_MODEL="${AUDIO_FLAMINGO_MODEL:-nvidia/audio-flamingo-3}"
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-100}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
export CACHE_ENABLED="${CACHE_ENABLED:-true}"
export WARMUP_ON_START="${WARMUP_ON_START:-true}"

# Optimization settings
export USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-true}"
export USE_MIXED_PRECISION="${USE_MIXED_PRECISION:-true}"
export PREFETCH_SIZE="${PREFETCH_SIZE:-3}"
export NUM_LOADING_THREADS="${NUM_LOADING_THREADS:-2}"

# Logging
export LOG_LEVEL="${LOG_LEVEL:-info}"

echo "Starting Optimized Audio Flamingo Server..."
echo "================================================"
echo "Model: $AUDIO_FLAMINGO_MODEL"
echo "Device: $DEVICE"
echo "Server: http://$HOST:$PORT"
echo "================================================"
echo "Optimizations:"
echo "  - Torch Compile: $USE_TORCH_COMPILE"
echo "  - Mixed Precision: $USE_MIXED_PRECISION"
echo "  - Prefetch Size: $PREFETCH_SIZE"
echo "  - Loading Threads: $NUM_LOADING_THREADS"
echo "  - Cache: $CACHE_ENABLED"
echo "================================================"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "WARNING: No conda environment detected. Activating 'data' environment..."
    conda activate data
fi

# Install server requirements if needed
if [ ! -f ".server_deps_installed" ]; then
    echo "Installing server dependencies..."
    pip install -r requirements_server.txt
    touch .server_deps_installed
fi

# Check GPU availability
if [ "$DEVICE" == "cuda" ]; then
    if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "ERROR: CUDA device requested but not available!"
        echo "Please run on a GPU node or set DEVICE=cpu"
        exit 1
    fi
    echo "GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# Start the server
if [ "$1" == "production" ]; then
    # Production mode with gunicorn
    echo "Starting in production mode with gunicorn..."
    gunicorn audio_flamingo_server_optimized:app \
        --workers 1 \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind $HOST:$PORT \
        --timeout 600 \
        --access-logfile - \
        --error-logfile - \
        --log-level $LOG_LEVEL
elif [ "$1" == "benchmark" ]; then
    # Benchmark mode with detailed logging
    echo "Starting in benchmark mode with detailed performance logging..."
    export PYTHONUNBUFFERED=1
    export TORCH_LOGS="+dynamo"
    export TORCHDYNAMO_VERBOSE=1
    python audio_flamingo_server_optimized.py
else
    # Development mode with auto-reload
    echo "Starting in development mode..."
    python audio_flamingo_server_optimized.py
fi