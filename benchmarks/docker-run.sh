#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Helper script to run benchmarks using Docker                               #
# ---------------------------------------------------------------------------- #

set -e

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run ServerlessLLM benchmarks using Docker container.

OPTIONS:
  -m, --model-name NAME        Model to benchmark (default: facebook/opt-6.7b)
  -n, --num-replicas N         Number of test replicas (default: 30)
  -p, --mem-pool-size SIZE     Memory pool size (default: 32GB)
  -s, --storage-path PATH      Host storage directory (default: /mnt/nvme)
  -r, --results-path PATH      Host results directory (default: ./results)
  -i, --image IMAGE            Docker image to use (default: serverlessllm/sllm:latest)
  -g, --gpu-limit LIMIT        GPU limit: 1, 2, "all", or "device=N" (default: 1)
  -t, --benchmark-type TYPE    Test type: random|single (default: random)
      --generate-plots         Generate visualization plots
      --keep-alive             Keep container running after completion
  -h, --help                   Show this help message

EXAMPLES:
  # Basic usage with custom model
  $0 --model-name meta-llama/Meta-Llama-3-8B

  # Specify GPU and replicas
  $0 --gpu-limit 2 --num-replicas 50

  # Use specific GPU device
  $0 --gpu-limit '"device=7"' --model-name deepseek-ai/DeepSeek-OCR

  # Backward compatible (env vars still work)
  MODEL_NAME=facebook/opt-6.7b NUM_REPLICAS=30 $0

For more information, see: benchmarks/README.md
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-name)
            CLI_MODEL_NAME="$2"
            shift 2
            ;;
        -n|--num-replicas)
            CLI_NUM_REPLICAS="$2"
            shift 2
            ;;
        -p|--mem-pool-size)
            CLI_MEM_POOL_SIZE="$2"
            shift 2
            ;;
        -s|--storage-path)
            CLI_STORAGE_PATH="$2"
            shift 2
            ;;
        -r|--results-path)
            CLI_RESULTS_PATH="$2"
            shift 2
            ;;
        -i|--image)
            CLI_IMAGE="$2"
            shift 2
            ;;
        -g|--gpu-limit)
            CLI_GPU_LIMIT="$2"
            shift 2
            ;;
        -t|--benchmark-type)
            CLI_BENCHMARK_TYPE="$2"
            shift 2
            ;;
        --generate-plots)
            CLI_GENERATE_PLOTS="true"
            shift
            ;;
        --keep-alive)
            CLI_KEEP_ALIVE="true"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Configuration: CLI args > environment variables > defaults
MODEL_NAME="${CLI_MODEL_NAME:-${MODEL_NAME:-facebook/opt-6.7b}}"
NUM_REPLICAS="${CLI_NUM_REPLICAS:-${NUM_REPLICAS:-30}}"
MEM_POOL_SIZE="${CLI_MEM_POOL_SIZE:-${MEM_POOL_SIZE:-32GB}}"
STORAGE_PATH="${CLI_STORAGE_PATH:-${STORAGE_PATH:-/mnt/nvme}}"
RESULTS_PATH="${CLI_RESULTS_PATH:-${RESULTS_PATH:-$(pwd)/results}}"
IMAGE="${CLI_IMAGE:-${IMAGE:-serverlessllm/sllm:latest}}"
GPU_LIMIT="${CLI_GPU_LIMIT:-${GPU_LIMIT:-1}}"
BENCHMARK_TYPE="${CLI_BENCHMARK_TYPE:-${BENCHMARK_TYPE:-random}}"
GENERATE_PLOTS="${CLI_GENERATE_PLOTS:-${GENERATE_PLOTS:-false}}"
KEEP_ALIVE="${CLI_KEEP_ALIVE:-${KEEP_ALIVE:-false}}"

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

# Build run-benchmark.sh arguments
BENCHMARK_ARGS=(
    --model-name "$MODEL_NAME"
    --num-replicas "$NUM_REPLICAS"
    --mem-pool-size "$MEM_POOL_SIZE"
    --storage-path /models
    --results-path /results
    --benchmark-type "$BENCHMARK_TYPE"
)

# Add boolean flags if set
if [ "$GENERATE_PLOTS" = "true" ]; then
    BENCHMARK_ARGS+=(--generate-plots)
fi

if [ "$KEEP_ALIVE" = "true" ]; then
    BENCHMARK_ARGS+=(--keep-alive)
fi

# Run benchmark (passing args as CLI flags instead of env vars)
docker run --rm --gpus "$GPU_LIMIT" \
    -e MODE=WORKER \
    -v "$STORAGE_PATH":/models \
    -v "$RESULTS_PATH":/results \
    -v "$SCRIPT_DIR":/scripts \
    --entrypoint /bin/bash \
    "$IMAGE" \
    /scripts/run-benchmark.sh "${BENCHMARK_ARGS[@]}"

echo ""
echo "=== Benchmark Complete ==="
echo "Results saved to: $RESULTS_PATH"
echo ""
echo "View summary:"
echo "  cat $RESULTS_PATH/summary.txt"
