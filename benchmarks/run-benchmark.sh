#!/bin/bash
# ---------------------------------------------------------------------------- #
#  ServerlessLLM Automated Benchmark Runner                                   #
# ---------------------------------------------------------------------------- #

set -e

# Configuration (via environment variables)
MODEL_NAME="${MODEL_NAME:-facebook/opt-1.3b}"
NUM_REPLICAS="${NUM_REPLICAS:-10}"
MEM_POOL_SIZE="${MEM_POOL_SIZE:-8GB}"
STORAGE_PATH="${STORAGE_PATH:-/models}"
RESULTS_PATH="${RESULTS_PATH:-/results}"
BENCHMARK_TYPE="${BENCHMARK_TYPE:-random}"
GENERATE_PLOTS="${GENERATE_PLOTS:-false}"
KEEP_ALIVE="${KEEP_ALIVE:-false}"

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

# Activate conda environment (required for official serverlessllm/sllm image)
log "Activating conda environment..."
source /opt/conda/etc/profile.d/conda.sh
conda activate worker || conda activate head
log "Conda environment activated: $(conda info --envs | grep '*' | awk '{print $1}')"

# Install dependencies if needed
log "Installing benchmark dependencies..."
pip install -q seaborn matplotlib pandas sentencepiece 2>&1 | tee -a "$LOG_FILE"

# NVMe detection (if script exists)
if [ -f "/scripts/detect-nvme.sh" ]; then
    log "Running NVMe detection..."
    bash /scripts/detect-nvme.sh 2>&1 | tee -a "$LOG_FILE" || log "NVMe detection failed (non-critical)"
fi

# Start sllm-store in background
log "Starting sllm-store server..."
sllm-store start \
    --storage-path "$STORAGE_PATH" \
    --mem-pool-size "$MEM_POOL_SIZE" \
    --chunk-size 16MB \
    --num-thread 4 \
    > "${RESULTS_PATH}/sllm-store.log" 2>&1 &

SLLM_STORE_PID=$!
log "sllm-store started (PID: $SLLM_STORE_PID)"

# Wait for sllm-store to be ready
log "Waiting for sllm-store to be ready..."
sleep 10

# Check if sllm-store is running
if ! kill -0 $SLLM_STORE_PID 2>/dev/null; then
    log "ERROR: sllm-store failed to start"
    cat "${RESULTS_PATH}/sllm-store.log"
    exit 1
fi
log "sllm-store is ready"

# Function to cleanup on exit
cleanup() {
    log "Cleaning up..."
    if [ -n "$SLLM_STORE_PID" ] && kill -0 $SLLM_STORE_PID 2>/dev/null; then
        log "Stopping sllm-store (PID: $SLLM_STORE_PID)"
        kill $SLLM_STORE_PID || true
        wait $SLLM_STORE_PID 2>/dev/null || true
    fi
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

    log "Downloading $NUM_REPLICAS replicas..."
    python3 download_models.py \
        --model-name "$MODEL_NAME" \
        --save-format "$MODEL_FORMAT" \
        --save-dir "$STORAGE_PATH" \
        --num-replicas "$NUM_REPLICAS" \
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
}

# Run each format in separate subprocess (GPU memory freed on subprocess exit)
for MODEL_FORMAT in safetensors sllm torch; do
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
