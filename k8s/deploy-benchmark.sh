#!/bin/bash
# ---------------------------------------------------------------------------- #
#  EIDF Benchmark Setup - Auto-configure from environment                     #
# ---------------------------------------------------------------------------- #

set -e

# Configuration from environment
NS="${NS:-}"
CPU="${CPU:-8}"
MEMORY="${MEMORY:-128Gi}"
GPU="${GPU:-1}"
NVME_PATH="${NVME_PATH:-/nvme}"
MODEL_NAME="${MODEL_NAME:-facebook/opt-1.3b}"
NUM_REPLICAS="${NUM_REPLICAS:-10}"

# Validate namespace
if [ -z "$NS" ]; then
    echo "ERROR: NS environment variable not set"
    echo "Usage: NS=<your-namespace> ./deploy-benchmark.sh"
    echo ""
    echo "Find your namespace:"
    echo "  kubectl get namespaces"
    exit 1
fi

echo "=== EIDF Benchmark Deployment ==="
echo "Namespace: $NS"
echo "Resources: ${CPU} CPU, ${MEMORY} RAM, ${GPU} GPU"
echo "NVMe Path: $NVME_PATH"
echo "Model: $MODEL_NAME (${NUM_REPLICAS} replicas)"
echo ""

# Create temp directory for processed manifests
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Processing manifests..."

# Function to process YAML with env var substitution
process_yaml() {
    local input=$1
    local output=$2

    # Always use sed (envsubst doesn't handle <PLACEHOLDER> syntax)
    sed -e "s|<YOUR_NAMESPACE>|${NS}|g" \
        -e "s|<CPU_REQUEST>|${CPU}|g" \
        -e "s|<CPU_LIMIT>|${CPU}|g" \
        -e "s|<MEMORY_REQUEST>|${MEMORY}|g" \
        -e "s|<MEMORY_LIMIT>|${MEMORY}|g" \
        -e "s|<GPU_COUNT>|${GPU}|g" \
        -e "s|<NVME_PATH>|${NVME_PATH}|g" \
        "$input" > "$output"
}

# Process each manifest
K8S_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "1. Processing benchmark-configmap.yaml..."
process_yaml "$K8S_DIR/benchmark-configmap.yaml" "$TEMP_DIR/benchmark-configmap.yaml"

echo "2. Processing benchmark-scripts-configmap.yaml..."
process_yaml "$K8S_DIR/benchmark-scripts-configmap.yaml" "$TEMP_DIR/benchmark-scripts-configmap.yaml"

echo "3. Processing benchmark-job-eidf.yaml..."
process_yaml "$K8S_DIR/benchmark-job-eidf.yaml" "$TEMP_DIR/benchmark-job-eidf.yaml"

# Apply configurations
echo ""
echo "Applying configurations..."

echo "  - benchmark-configmap"
kubectl apply -f "$TEMP_DIR/benchmark-configmap.yaml" -n "$NS"

echo "  - benchmark-scripts"
kubectl apply -f "$TEMP_DIR/benchmark-scripts-configmap.yaml" -n "$NS"

echo ""
echo "Creating benchmark job..."
kubectl create -f "$TEMP_DIR/benchmark-job-eidf.yaml" -n "$NS"

echo ""
echo "=== Deployment Complete ==="
echo ""

# Wait a moment for job to be created
sleep 2

# Get job name
JOB_NAME=$(kubectl get jobs -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")

if [ -n "$JOB_NAME" ]; then
    echo "Job created: $JOB_NAME"
    echo ""
    echo "Monitor logs:"
    echo "  kubectl logs -f job/$JOB_NAME -n $NS"
    echo ""
    echo "Check status:"
    echo "  kubectl get job/$JOB_NAME -n $NS"
    echo ""
    echo "Get results (after completion):"
    POD_NAME=$(kubectl get pods -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null || echo "")
    if [ -n "$POD_NAME" ]; then
        echo "  kubectl logs $POD_NAME -n $NS > benchmark-output.log"
        echo "  kubectl cp $NS/$POD_NAME:/results ./results"
    fi
else
    echo "Job created (name auto-generated)"
    echo ""
    echo "Find job:"
    echo "  kubectl get jobs -n $NS | grep sllm-benchmark"
fi

echo ""
echo "Processed manifests saved to: $TEMP_DIR"
echo "(Will be cleaned up on exit)"
