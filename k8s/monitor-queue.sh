#!/bin/bash
# ---------------------------------------------------------------------------- #
#  Monitor Kueue queue status on EIDF                                         #
# ---------------------------------------------------------------------------- #

set -e

# Help function
show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Monitor Kueue queue status for ServerlessLLM benchmarks in Kubernetes.

OPTIONS:
  -n, --namespace NS           Kubernetes namespace (required)
  -d, --detailed               Show detailed queue/workload information
  -h, --help                   Show this help message

EXAMPLES:
  # Basic queue status
  $0 --namespace my-namespace

  # Show detailed information
  $0 --namespace my-namespace --detailed

  # Backward compatible (env vars still work)
  NS=my-namespace $0
  NS=my-namespace DETAILED=true $0

For more information, see: k8s/QUEUE_MONITORING.md
EOF
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            CLI_NS="$2"
            shift 2
            ;;
        -d|--detailed)
            CLI_DETAILED="true"
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
NS="${CLI_NS:-${NS:-}}"
DETAILED="${CLI_DETAILED:-${DETAILED:-false}}"

if [ -z "$NS" ]; then
    echo "ERROR: Namespace not specified"
    echo ""
    show_help
    exit 1
fi

QUEUE="${NS}-user-queue"

echo "=== Kueue Queue Monitoring ==="
echo "Namespace: $NS"
echo "Queue: $QUEUE"
echo ""

# Check if job exists
JOB=$(kubectl get jobs -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1].metadata.name}' 2>/dev/null)

if [ -n "$JOB" ]; then
    echo "Latest Job: $JOB"
    echo ""
fi

# Queue status (summary)
echo "=== Queue Status ==="
kubectl get queue "$QUEUE" -n "$NS" 2>/dev/null || echo "Queue not found"
echo ""

# Workloads (summary)
echo "=== Workloads in Queue ==="
kubectl get workloads -n "$NS" 2>/dev/null || echo "No workloads found"
echo ""

# If job exists, show workload status
if [ -n "$JOB" ]; then
    WORKLOAD=$(kubectl get workloads -n "$NS" -o jsonpath="{.items[?(@.metadata.ownerReferences[0].name=='$JOB')].metadata.name}" 2>/dev/null)

    if [ -n "$WORKLOAD" ]; then
        echo "=== Current Job Workload Status ==="
        kubectl get workload "$WORKLOAD" -n "$NS" 2>/dev/null
        echo ""
    fi
fi

# Only show detailed info if requested
if [ "$DETAILED" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "=== DETAILED INFORMATION ==="
    echo "=========================================="
    echo ""

    # Detailed queue info
    echo "=== Queue Details ==="
    kubectl describe queue "$QUEUE" -n "$NS" 2>/dev/null || echo "Queue not found"
    echo ""

    # Detailed workload info
    if [ -n "$JOB" ] && [ -n "$WORKLOAD" ]; then
        echo "=== Workload Details for Job: $JOB ==="
        kubectl describe workload "$WORKLOAD" -n "$NS"
    fi
else
    echo "Tip: For detailed queue/workload info, run:"
    echo "  $0 --namespace $NS --detailed"
fi
