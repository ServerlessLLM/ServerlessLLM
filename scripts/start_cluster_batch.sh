#!/bin/bash
set -e

# === Start Cluster with Batch Processing ENABLED ===
# This script starts the full ServerlessLLM stack with the Batch Scheduler active.
# Use this for testing batch jobs, DAG dependencies, and scheduler optimizations.

# MPS Workaround (User specific)
export CUDA_MPS_PIPE_DIRECTORY=/tmp/no_mps_vllm

# Fix: CUDA toolkit stub shadows real driver (since CUDA toolkit update 2026-03-12)
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

# Activate virtual environment explicitly
source .venv_new/bin/activate

# Configuration
PYTHON=.venv_new/bin/python
PYLET_BIN=.venv_new/bin/pylet
SLLM_BIN=.venv_new/bin/sllm
PORT=8343

echo "=== GPU Configuration ==="
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES: Not Set (Using all available GPUs)"
fi

echo "Visible GPUs (as seen by PyTorch):"
$PYTHON -c "import torch; print(f'Count: {torch.cuda.device_count()}'); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" || echo "Failed to query PyTorch."
echo "========================="

# Paths (using distinct directories for isolation)
MODELS_DIR="./models_batch"
DB_PATH="./state_batch.db"

# Cleanup
echo "Cleaning up previous run..."
pkill -f "pylet start" || true
pkill -f "sllm start" || true
pkill -f "sllm-store" || true
sleep 2

mkdir -p $MODELS_DIR
rm -f ${DB_PATH}*
rm -f ~/.pylet/pylet.db*

echo "=== 1. Starting Pylet Head (Cluster Manager) ==="
$PYLET_BIN start > pylet_head_batch.log 2>&1 &
echo "Pylet Head PID: $!"
sleep 2

# Auto-detect GPU count from CUDA_VISIBLE_DEVICES
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
else
    GPU_COUNT=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
echo "=== 2. Starting Pylet Worker ($GPU_COUNT GPUs) ==="
# Connects to localhost:8000
# We explicitly pass the env var again to be absolutely sure
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYLET_BIN start --head localhost:8000 --gpu-units $GPU_COUNT > pylet_worker_batch.log 2>&1 &
echo "Pylet Worker PID: $!"
sleep 2

echo "=== 3. Starting SLLM Head (Gateway + Router + Batch Scheduler) ==="
# Explicitly enable batch scheduler
export ENABLE_BATCH_SCHEDULER=true
# Pass pinned memory pool size to sllm-store (80 GB fits Qwen2.5-32B-Instruct ~64 GB)
export SLLM_STORE_MEM_POOL_SIZE=80GB

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $SLLM_BIN start \
    --host 0.0.0.0 \
    --port $PORT \
    --pylet-endpoint http://localhost:8000 \
    --database-path "$DB_PATH" \
    --storage-path "$MODELS_DIR" > sllm_head_batch.log 2>&1 &
echo "SLLM Head PID: $!"

echo ""
echo "============================================================"
echo "Cluster Started (Batch Mode)"
echo "API Endpoint: http://localhost:$PORT"
echo "Batch Endpoint: http://localhost:$PORT/v1/batches"
echo "Logs: sllm_head_batch.log"
echo "============================================================"
echo "Press Ctrl+C to stop the cluster."

# === Example Usage (Run in another terminal) ===
echo ""
echo "=== To Submit a 10-Task Batch Job (Qwen 0.6B & 7B) ==="
echo "Run the following command in another terminal:"
echo ""
echo "curl -X POST http://localhost:8343/v1/batches \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"tasks\": ["
echo "      {\"custom_id\": \"task-1-small\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"1+1=?\"}], \"max_tokens\": 20}},"
echo "      {\"custom_id\": \"task-2-small\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Name a color.\"}], \"max_tokens\": 20}},"
echo "      {\"custom_id\": \"task-3-small\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of Italy?\"}], \"max_tokens\": 20}},"
echo "      {\"custom_id\": \"task-4-small\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a haiku about code.\"}], \"max_tokens\": 20}},"
echo "      {\"custom_id\": \"task-5-small\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen3-0.6B\", \"messages\": [{\"role\": \"user\", \"content\": \"Is Python a snake?\"}], \"max_tokens\": 20}},"
echo "      {\"custom_id\": \"task-6-big\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Explain quantum entanglement simply.\"}], \"max_tokens\": 50}},"
echo "      {\"custom_id\": \"task-7-big\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"List 3 benefits of exercise.\"}], \"max_tokens\": 50}},"
echo "      {\"custom_id\": \"task-8-big\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Who wrote Hamlet?\"}], \"max_tokens\": 50}},"
echo "      {\"custom_id\": \"task-9-big\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Translate Hello to Spanish.\"}], \"max_tokens\": 50}},"
echo "      {\"custom_id\": \"task-10-big\", \"method\": \"POST\", \"url\": \"/v1/chat/completions\", \"body\": {\"model\": \"Qwen/Qwen2.5-7B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the speed of light?\"}], \"max_tokens\": 50}}"
echo "    ]"
echo "  }'"
echo ""
echo "=== To Check Batch Status ==="
echo "curl http://localhost:8343/v1/batches/<BATCH_ID>"
echo ""

wait
