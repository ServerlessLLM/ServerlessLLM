#!/bin/bash
# ---------------------------------------------------------------------------- #
#  ServerlessLLM Automated Benchmark Runner                                   #
# ---------------------------------------------------------------------------- #

set -e

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run ServerlessLLM benchmarks with automated model loading and testing.

OPTIONS:
  -m, --model-name NAME        Model to benchmark (default: facebook/opt-6.7b)
  -n, --num-replicas N         Number of test replicas (default: 30)
  -p, --mem-pool-size SIZE     Memory pool size (default: 32GB)
  -s, --storage-path PATH      Model storage directory (default: ./models)
  -r, --results-path PATH      Results output directory (default: ./results)
  -t, --benchmark-type TYPE    Test type: random|cached (default: random)
      --generate-plots         Generate visualization plots
      --keep-alive             Keep container running after completion
  -h, --help                   Show this help message

EXAMPLES:
  # Basic usage with custom model
  $0 --model-name meta-llama/Meta-Llama-3-8B

  # Full configuration
  $0 --model-name facebook/opt-6.7b \\
     --num-replicas 50 \\
     --mem-pool-size 64GB \\
     --generate-plots

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
STORAGE_PATH="${CLI_STORAGE_PATH:-${STORAGE_PATH:-./models}}"
RESULTS_PATH="${CLI_RESULTS_PATH:-${RESULTS_PATH:-./results}}"
BENCHMARK_TYPE="${CLI_BENCHMARK_TYPE:-${BENCHMARK_TYPE:-random}}"
GENERATE_PLOTS="${CLI_GENERATE_PLOTS:-${GENERATE_PLOTS:-false}}"
KEEP_ALIVE="${CLI_KEEP_ALIVE:-${KEEP_ALIVE:-false}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${RESULTS_PATH}/benchmark.log"
SUMMARY_FILE="${RESULTS_PATH}/summary.txt"

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Create results directory
mkdir -p "$RESULTS_PATH"

log "=== ServerlessLLM Automated Benchmark ==="
log "Model: $MODEL_NAME"
log "Replicas: $NUM_REPLICAS"
log "Memory Pool: $MEM_POOL_SIZE"
log "Storage: $STORAGE_PATH"
log "Benchmark Type: $BENCHMARK_TYPE"
log ""

# Activate conda environment if running in Docker (serverlessllm/sllm image)
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    log "Activating conda environment..."
    source /opt/conda/etc/profile.d/conda.sh
    conda activate worker || conda activate head || true
    log "Conda environment: $(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}' || echo 'default')"

    # Install dependencies in Docker
    log "Installing benchmark dependencies..."
    pip install -q seaborn matplotlib pandas sentencepiece 2>&1 | tee -a "$LOG_FILE" || true
fi

# Global variable for sllm-store PID
SLLM_STORE_PID=""

# Function to start sllm-store
start_sllm_store() {
    log "Starting sllm-store server..."
    sllm-store start \
        --storage-path "$STORAGE_PATH" \
        --mem-pool-size "$MEM_POOL_SIZE" \
        --chunk-size 16MB \
        --num-thread 4 \
        > "${RESULTS_PATH}/sllm-store.log" 2>&1 &

    SLLM_STORE_PID=$!
    log "sllm-store started (PID: $SLLM_STORE_PID)"

    # Wait for sllm-store to be ready (up to 300 seconds)
    log "Waiting for sllm-store to be ready (pinning $MEM_POOL_SIZE memory)..."
    local elapsed=0
    local timeout=300
    local interval=2

    while [ $elapsed -lt $timeout ]; do
        # Check if process is still running
        if ! kill -0 $SLLM_STORE_PID 2>/dev/null; then
            log "ERROR: sllm-store process died during startup"
            cat "${RESULTS_PATH}/sllm-store.log"
            exit 1
        fi

        # Check if sllm-store is ready by looking for success message in log
        if grep -q "Starting gRPC server on" "${RESULTS_PATH}/sllm-store.log" 2>/dev/null; then
            log "sllm-store is ready (took ${elapsed}s)"
            return 0
        fi

        sleep $interval
        elapsed=$((elapsed + interval))

        # Log progress every 30 seconds
        if [ $((elapsed % 30)) -eq 0 ]; then
            log "Still waiting for sllm-store... (${elapsed}s elapsed)"
        fi
    done

    # Timeout reached
    log "ERROR: sllm-store failed to become ready within ${timeout}s"
    log "Last log output:"
    tail -50 "${RESULTS_PATH}/sllm-store.log"
    exit 1
}

# Function to stop sllm-store
stop_sllm_store() {
    if [ -n "$SLLM_STORE_PID" ] && kill -0 $SLLM_STORE_PID 2>/dev/null; then
        log "Stopping sllm-store (PID: $SLLM_STORE_PID)"
        kill $SLLM_STORE_PID || true
        wait $SLLM_STORE_PID 2>/dev/null || true
        SLLM_STORE_PID=""
    fi
}

# Function to cleanup on exit
cleanup() {
    log "Cleaning up..."
    stop_sllm_store
}

trap cleanup EXIT INT TERM

# Run benchmarks
log "Starting benchmarks..."
cd "$SCRIPT_DIR"

# Clean storage directory
log "Cleaning storage directory..."
rm -rf "${STORAGE_PATH:?}"/* || true

# Run each format in subprocess to ensure GPU memory is freed
run_format_benchmark() {
    local MODEL_FORMAT=$1
    log "=== Testing $MODEL_FORMAT format ==="

    # Start sllm-store only for sllm format (saves memory for safetensors benchmark)
    if [ "$MODEL_FORMAT" = "sllm" ]; then
        start_sllm_store
    fi

    # For cached tests, only need 1 replica (loaded repeatedly)
    local DOWNLOAD_REPLICAS=$NUM_REPLICAS
    if [ "$BENCHMARK_TYPE" = "cached" ]; then
        DOWNLOAD_REPLICAS=1
        log "Cached test: downloading 1 replica (will be loaded $NUM_REPLICAS times)..."
    else
        log "Downloading $NUM_REPLICAS replicas..."
    fi

    python3 download_models.py \
        --model-name "$MODEL_NAME" \
        --save-format "$MODEL_FORMAT" \
        --save-dir "$STORAGE_PATH" \
        --num-replicas "$DOWNLOAD_REPLICAS" \
        2>&1 | tee -a "$LOG_FILE"

    log "Running benchmark..."
    python3 test_loading.py \
        --model-name "$MODEL_NAME" \
        --model-format "$MODEL_FORMAT" \
        --model-dir "$STORAGE_PATH" \
        --num-replicas "$NUM_REPLICAS" \
        --benchmark-type "$BENCHMARK_TYPE" \
        --output-dir "$RESULTS_PATH" \
        2>&1 | tee -a "$LOG_FILE"

    log "Cleaning storage..."
    rm -rf "${STORAGE_PATH:?}"/* || true

    # Stop sllm-store after sllm format benchmark (frees memory)
    if [ "$MODEL_FORMAT" = "sllm" ]; then
        stop_sllm_store
    fi
}

# Run each format in separate subprocess (GPU memory freed on subprocess exit)
for MODEL_FORMAT in safetensors sllm; do
    ( run_format_benchmark "$MODEL_FORMAT" )
    log "Format $MODEL_FORMAT completed, GPU memory released"
    sleep 5
    log ""
done

# Generate summary report
log "Generating benchmark report..."
if [ -f "$SCRIPT_DIR/generate_report.py" ]; then
    python3 "$SCRIPT_DIR/generate_report.py" \
        --model-name "$MODEL_NAME" \
        --num-replicas "$NUM_REPLICAS" \
        --benchmark-type "$BENCHMARK_TYPE" \
        --results-dir "$RESULTS_PATH" \
        --output-file "$SUMMARY_FILE" \
        --generate-plots "$GENERATE_PLOTS" \
        2>&1 | tee -a "$LOG_FILE"

    log ""
    log "=== Benchmark Summary ==="
    cat "$SUMMARY_FILE" | tee -a "$LOG_FILE"
else
    log "Warning: generate_report.py not found, skipping summary"
fi

log ""
log "=== Benchmark Complete ==="
log "Results saved to: $RESULTS_PATH"
log "Summary: $SUMMARY_FILE"
log "Full log: $LOG_FILE"

# Keep container alive if requested
if [ "$KEEP_ALIVE" = "true" ]; then
    log "KEEP_ALIVE=true, keeping container running..."
    log "Press Ctrl+C to exit"
    tail -f /dev/null
fi
