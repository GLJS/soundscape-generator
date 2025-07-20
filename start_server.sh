#!/bin/bash

# Audio Flamingo HTTP Server Startup Script
# This script starts the FastAPI server for Audio Flamingo inference

# Configuration
export AUDIO_FLAMINGO_MODEL="${AUDIO_FLAMINGO_MODEL:-nvidia/audio-flamingo-3}"
export DEVICE="${DEVICE:-cuda}"
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-100}"
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
export CACHE_ENABLED="${CACHE_ENABLED:-true}"
export WARMUP_ON_START="${WARMUP_ON_START:-true}"

# Logging
export LOG_LEVEL="${LOG_LEVEL:-info}"

echo "Starting Audio Flamingo Server..."
echo "Model: $AUDIO_FLAMINGO_MODEL"
echo "Device: $DEVICE"
echo "Server: http://$HOST:$PORT"
echo "Queue Size: $MAX_QUEUE_SIZE"
echo "Cache: $CACHE_ENABLED"

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

# Start the server
if [ "$1" == "production" ]; then
    # Production mode with gunicorn
    echo "Starting in production mode with gunicorn..."
    gunicorn audio_flamingo_server:app \
        --workers 1 \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind $HOST:$PORT \
        --timeout 600 \
        --access-logfile - \
        --error-logfile - \
        --log-level $LOG_LEVEL
else
    # Development mode with uvicorn auto-reload
    echo "Starting in development mode with auto-reload..."
    python audio_flamingo_server.py
fi