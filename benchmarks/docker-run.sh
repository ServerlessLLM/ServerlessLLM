#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Helper script to run benchmarks using Docker                               #
# ---------------------------------------------------------------------------- #

set -e

# Default configuration
MODEL_NAME="${MODEL_NAME:-facebook/opt-1.3b}"
NUM_REPLICAS="${NUM_REPLICAS:-10}"
MEM_POOL_SIZE="${MEM_POOL_SIZE:-8GB}"
STORAGE_PATH="${STORAGE_PATH:-/mnt/nvme}"
RESULTS_PATH="${RESULTS_PATH:-$(pwd)/results}"
IMAGE="${IMAGE:-serverlessllm/sllm:latest}"

echo "=== ServerlessLLM Benchmark (Docker) ==="
echo "Image: $IMAGE"
echo "Model: $MODEL_NAME"
echo "Replicas: $NUM_REPLICAS"
echo "Storage: $STORAGE_PATH"
echo "Results: $RESULTS_PATH"
echo ""

# Create results directory
mkdir -p "$RESULTS_PATH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run benchmark
docker run --rm --gpus all \
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
