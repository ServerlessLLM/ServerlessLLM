#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Helper script to run benchmarks using Docker                               #
# ---------------------------------------------------------------------------- #

set -e

# Default configuration
MODEL_NAME="${MODEL_NAME:-facebook/opt-6.7b}"
NUM_REPLICAS="${NUM_REPLICAS:-30}"
MEM_POOL_SIZE="${MEM_POOL_SIZE:-32GB}"
STORAGE_PATH="${STORAGE_PATH:-/mnt/nvme}"
RESULTS_PATH="${RESULTS_PATH:-$(pwd)/results}"
IMAGE="${IMAGE:-serverlessllm/sllm:latest}"
# GPU_LIMIT options: 1, 2, "all", '"device=0"', '"device=0,1"', etc.
GPU_LIMIT="${GPU_LIMIT:-1}"

echo "=== ServerlessLLM Benchmark (Docker) ==="
echo "Image: $IMAGE"
echo "Model: $MODEL_NAME"
echo "Replicas: $NUM_REPLICAS"
echo "GPU Limit: $GPU_LIMIT"
echo "Storage: $STORAGE_PATH"
echo "Results: $RESULTS_PATH"
echo ""

# Create results directory
mkdir -p "$RESULTS_PATH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run benchmark
docker run --rm --gpus "$GPU_LIMIT" \
    -e MODE=WORKER \
    -e MODEL_NAME="$MODEL_NAME" \
    -e NUM_REPLICAS="$NUM_REPLICAS" \
    -e MEM_POOL_SIZE="$MEM_POOL_SIZE" \
    -e STORAGE_PATH=/models \
    -e RESULTS_PATH=/results \
    -v "$STORAGE_PATH":/models \
    -v "$RESULTS_PATH":/results \
    -v "$SCRIPT_DIR":/scripts \
    --entrypoint /bin/bash \
    "$IMAGE" \
    /scripts/run-benchmark.sh

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $RESULTS_PATH"
echo ""
echo "View summary:"
echo "  cat $RESULTS_PATH/summary.txt"
