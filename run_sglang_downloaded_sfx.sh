#!/bin/bash
# Run Downloaded SFX with SGLang server

# Check if on GPU node
if [[ $(hostname) == gcn* ]]; then
    echo "On GPU node, proceeding..."
else
    echo "Not on GPU node. Please run with:"
    echo "srun --partition=gpu_h100 --time=2:00:00 --gpus-per-node=1 --exclude=gcn131,gcn138 bash run_sglang_downloaded_sfx.sh"
    exit 1
fi

# Activate conda environment
source ~/.bashrc
conda activate data

module load CUDA/12.6.0

# Start SGLang server
echo "Starting SGLang server..."
python -m sglang.launch_server --model-path google/gemma-3n-e4b-it --attention-backend fa3 --port 30000 > sglang_server.log 2>&1 &
SGLANG_PID=$!

# Wait for server
echo "Waiting for server to start..."
for i in {1..120}; do
    if curl -s http://127.0.0.1:30000/v1/models > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 120 ]; then
        echo "Server failed to start"
        cat sglang_server.log
        exit 1
    fi
    sleep 1
done

# Run processor
echo "Running Downloaded SFX processor..."
python laion_datasets/downloaded_sfx_sglang.py \
    --audio-dir /gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/extracted \
    --metadata /gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx \
    --output-dir /scratch-shared/gwijngaard/tar/downloaded_sfx_sglang \
    --samples-per-tar 2048 \
    --batch-size 128 \
    --num-workers 16

# Cleanup
echo "Stopping SGLang server..."
kill $SGLANG_PID
wait $SGLANG_PID 2>/dev/null

echo "Done!"