#!/bin/bash
# Monitor benchmark job progress

NS="${NS:-}"

if [ -z "$NS" ]; then
    echo "Usage: NS=<namespace> $0"
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
            echo "  NS=$NS k8s/monitor-queue.sh"
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
    echo "  NS=$NS k8s/monitor-queue.sh"
    exit 1
fi
