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
  -j, --job-name NAME          Specific job name to monitor (optional)
  -h, --help                   Show this help message

EXAMPLES:
  # Monitor latest benchmark job
  $0 --namespace my-namespace

  # Monitor specific job
  $0 --namespace my-namespace --job-name sllm-benchmark-abc123

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
        -j|--job-name)
            CLI_JOB_NAME="$2"
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
JOB_NAME="${CLI_JOB_NAME:-${JOB_NAME:-}}"

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

# List all available jobs
echo "=== Available Benchmark Jobs ==="
ALL_JOBS=$(kubectl get jobs -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o custom-columns=NAME:.metadata.name,STATUS:.status.conditions[0].type,COMPLETIONS:.status.succeeded,AGE:.metadata.creationTimestamp 2>/dev/null)

if [ -z "$ALL_JOBS" ]; then
    echo "No benchmark jobs found in namespace $NS"
    exit 1
fi

echo "$ALL_JOBS"
echo ""

# Select job to monitor
if [ -n "$JOB_NAME" ]; then
    # User specified a job name
    JOB="$JOB_NAME"
    echo "Monitoring specified job: $JOB"
else
    # Find the most recent job (sort by creation time, get the newest)
    # Using tail to get the last line after sorting (most recent)
    JOB=$(kubectl get jobs -n "$NS" -l app=sllm-benchmark --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[*].metadata.name}' 2>/dev/null | tr ' ' '\n' | tail -1)

    if [ -z "$JOB" ]; then
        echo "ERROR: Could not determine latest job"
        exit 1
    fi

    echo "Monitoring latest job: $JOB"
fi

# Verify job exists
if ! kubectl get job "$JOB" -n "$NS" &>/dev/null; then
    echo "ERROR: Job '$JOB' not found in namespace $NS"
    echo ""
    echo "Available jobs:"
    kubectl get jobs -n "$NS" -l app=sllm-benchmark -o name
    exit 1
fi

# Show job details
echo ""
echo "=== Job Details ==="
kubectl get job "$JOB" -n "$NS"
echo ""

# Find pod for this specific job
echo "=== Finding Pod for Job ==="
POD=$(kubectl get pods -n "$NS" -l app=sllm-benchmark,job-name="$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)

if [ -z "$POD" ]; then
    echo "Waiting for pod to be created..."
    sleep 5
    POD=$(kubectl get pods -n "$NS" -l app=sllm-benchmark,job-name="$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
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
