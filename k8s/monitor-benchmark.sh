#!/bin/bash
# Monitor benchmark job progress

NS="${NS:-}"

if [ -z "$NS" ]; then
    echo "Usage: NS=<namespace> $0"
    exit 1
fi

echo "Monitoring benchmark in namespace: $NS"
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
    echo ""
    echo "=== Following logs (Ctrl+C to stop) ==="
    echo ""
    kubectl logs -f "$POD" -n "$NS"
else
    echo "ERROR: Could not find pod"
    exit 1
fi
