#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Monitor ServerlessLLM benchmark job progress                                #
# ---------------------------------------------------------------------------- #

set -e

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Monitor ServerlessLLM benchmark job progress in Kubernetes.

OPTIONS:
  -n, --namespace NS           Kubernetes namespace (required)
  -h, --help                   Show this help message

EXAMPLES:
  # Monitor benchmark in a namespace
  $0 --namespace my-namespace

  # Backward compatible (env vars still work)
  NS=my-namespace $0

For more information, see: k8s/README.md
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            CLI_NS="$2"
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

if [ -z "$NS" ]; then
    echo "ERROR: Namespace not specified"
    echo ""
    show_help
    exit 1
fi

echo "Monitoring benchmark in namespace: $NS"
echo ""

# Check queue status first
echo "=== Queue Status ==="
kubectl get queue "${NS}-user-queue" -n "$NS" 2>/dev/null || echo "Queue info not available"
echo ""

# Find latest job
JOB=$(kubectl get jobs -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)

if [ -z "$JOB" ]; then
    echo "No benchmark jobs found in namespace $NS"
    exit 1
fi

echo "Job: $JOB"

# Find pod
POD=$(kubectl get pods -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)

if [ -z "$POD" ]; then
    echo "Waiting for pod to be created..."
    sleep 5
    POD=$(kubectl get pods -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)
fi

if [ -n "$POD" ]; then
    echo "Pod: $POD"

    # Check pod status
    POD_STATUS=$(kubectl get pod "$POD" -n "$NS" -o jsonpath='{.status.phase}' 2>/dev/null)
    echo "Status: $POD_STATUS"

    if [ "$POD_STATUS" = "Pending" ]; then
        echo ""
        echo "Pod is pending. Checking workload status..."
        WORKLOAD=$(kubectl get workloads -n "$NS" -o jsonpath="{.items[?(@.metadata.ownerReferences[0].name=='$JOB')].metadata.name}" 2>/dev/null)
        if [ -n "$WORKLOAD" ]; then
            echo "Workload: $WORKLOAD"
            kubectl get workload "$WORKLOAD" -n "$NS" 2>/dev/null
            echo ""
            echo "Tip: Job may be queued. Check queue status:"
            echo "  k8s/monitor-queue.sh --namespace $NS"
        fi
        echo ""
        echo "Waiting for pod to start..."
        kubectl wait --for=condition=Ready pod/"$POD" -n "$NS" --timeout=300s || true
    fi

    echo ""
    echo "=== Following logs (Ctrl+C to stop) ==="
    echo ""
    kubectl logs -f "$POD" -n "$NS"
else
    echo "ERROR: Could not find pod"
    echo ""
    echo "Check if job is queued:"
    echo "  k8s/monitor-queue.sh --namespace $NS"
    exit 1
fi
