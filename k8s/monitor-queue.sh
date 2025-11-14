#!/bin/bash
# Monitor Kueue queue status on EIDF

NS="${NS:-}"
DETAILED="${DETAILED:-false}"

if [ -z "$NS" ]; then
    echo "Usage: NS=<namespace> $0"
    echo "       NS=<namespace> DETAILED=true $0  # Show full details"
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
    echo "  NS=$NS DETAILED=true $0"
fi
