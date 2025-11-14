#!/bin/bash
# Monitor Kueue queue status on EIDF

NS="${NS:-}"

if [ -z "$NS" ]; then
    echo "Usage: NS=<namespace> $0"
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

# Queue status
echo "=== Queue Status ==="
kubectl get queue "$QUEUE" -n "$NS" 2>/dev/null || echo "Queue not found"
echo ""

# Workloads
echo "=== Workloads in Queue ==="
kubectl get workloads -n "$NS" 2>/dev/null || echo "No workloads found"
echo ""

# Detailed queue info
echo "=== Queue Details ==="
kubectl describe queue "$QUEUE" -n "$NS" 2>/dev/null || echo "Queue not found"
echo ""

# If job exists, show workload details
if [ -n "$JOB" ]; then
    WORKLOAD=$(kubectl get workloads -n "$NS" -o jsonpath="{.items[?(@.metadata.ownerReferences[0].name=='$JOB')].metadata.name}" 2>/dev/null)

    if [ -n "$WORKLOAD" ]; then
        echo "=== Workload Details for Job: $JOB ==="
        kubectl describe workload "$WORKLOAD" -n "$NS"
    fi
fi
