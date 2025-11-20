#!/bin/bash
# ---------------------------------------------------------------------------- #
#  EIDF Benchmark Setup - Deploy benchmarks to Kubernetes                     #
# ---------------------------------------------------------------------------- #

set -e

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy ServerlessLLM benchmarks to Kubernetes cluster.

OPTIONS:
  --namespace NS               Kubernetes namespace (required)
  --cpu CORES                  CPU cores (default: 8)
  --memory SIZE                Memory size (default: 128Gi)
  --gpu COUNT                  GPU count (default: 1)
  --nvme-path PATH             NVMe mount path (default: /nvme)
  -m, --model-name NAME        Model to benchmark (default: facebook/opt-6.7b)
  -n, --num-replicas N         Number of test replicas (default: 30)
  -h, --help                   Show this help message

EXAMPLES:
  # Basic deployment
  $0 --namespace my-namespace

  # Full configuration
  $0 --namespace my-namespace \\
     --cpu 16 \\
     --memory 256Gi \\
     --gpu 2 \\
     --model-name meta-llama/Meta-Llama-3-8B \\
     --num-replicas 50

  # Backward compatible (env vars still work)
  NS=my-namespace CPU=16 $0

For more information, see: k8s/README.md
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace)
            CLI_NS="$2"
            shift 2
            ;;
        --cpu)
            CLI_CPU="$2"
            shift 2
            ;;
        --memory)
            CLI_MEMORY="$2"
            shift 2
            ;;
        --gpu)
            CLI_GPU="$2"
            shift 2
            ;;
        --nvme-path)
            CLI_NVME_PATH="$2"
            shift 2
            ;;
        -m|--model-name)
            CLI_MODEL_NAME="$2"
            shift 2
            ;;
        -n|--num-replicas)
            CLI_NUM_REPLICAS="$2"
            shift 2
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
NS="${CLI_NS:-${NS:-}}"
CPU="${CLI_CPU:-${CPU:-8}}"
MEMORY="${CLI_MEMORY:-${MEMORY:-128Gi}}"
GPU="${CLI_GPU:-${GPU:-1}}"
NVME_PATH="${CLI_NVME_PATH:-${NVME_PATH:-/nvme}}"
MODEL_NAME="${CLI_MODEL_NAME:-${MODEL_NAME:-facebook/opt-6.7b}}"
NUM_REPLICAS="${CLI_NUM_REPLICAS:-${NUM_REPLICAS:-30}}"

# Validate namespace
if [ -z "$NS" ]; then
    echo "ERROR: Namespace not specified"
    echo ""
    show_help
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
        -e "s|<MODEL_NAME>|${MODEL_NAME}|g" \
        -e "s|<NUM_REPLICAS>|${NUM_REPLICAS}|g" \
        -e "s|<MEM_POOL_SIZE>|${MEM_POOL_SIZE}|g" \
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
