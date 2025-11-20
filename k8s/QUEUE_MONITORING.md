# Kueue Queue Monitoring on EIDF

EIDF uses Kueue for job scheduling and resource quota management.

## Quick Commands

```bash
# Monitor queue and workload status
k8s/monitor-queue.sh --namespace sc24029

# Monitor benchmark progress (includes queue status)
k8s/monitor-benchmark.sh --namespace sc24029

# Manual commands
kubectl get queue sc24029-user-queue -n sc24029
kubectl get workloads -n sc24029
```

## Understanding Job States

### 1. Job Submitted → 2. Queued → 3. Running → 4. Completed

**Check current state:**
```bash
kubectl get workloads -n sc24029
# QuotaReserved = waiting in queue
# Admitted = resources assigned, starting
```

## Queue Monitoring Commands

### Check Queue Status
```bash
kubectl get queue sc24029-user-queue -n sc24029
# Shows: PENDING WORKLOADS, ADMITTED WORKLOADS
```

### Detailed Queue Info
```bash
kubectl describe queue sc24029-user-queue -n sc24029
# Shows: resource usage, GPU flavors, quotas
```

### Check Your Workloads
```bash
kubectl get workloads -n sc24029
kubectl describe workload <workload-name> -n sc24029
```

## Common Scenarios

### Job Queued (Waiting)
```bash
$ kubectl get workloads -n sc24029
NAME                      STATUS          AGE
job-sllm-benchmark-xyz    QuotaReserved   2m

# Meaning: Waiting for GPU/resources
# Action: Wait or check queue details
```

### Job Admitted (Starting)
```bash
$ kubectl get workloads -n sc24029
NAME                      STATUS      AGE
job-sllm-benchmark-xyz    Admitted    3m

# Meaning: Resources allocated, pod starting
```

### Job Running
```bash
$ kubectl get pods -n sc24029
NAME                   STATUS    AGE
sllm-benchmark-xyz     Running   2m

# Follow logs
kubectl logs -f sllm-benchmark-xyz -n sc24029
```

### Job Stuck in Queue
```bash
kubectl describe queue sc24029-user-queue -n sc24029
# Check quota limits and cluster capacity
# Wait for other jobs or reduce resource requests
```

## Watch Commands (Auto-Refresh)

```bash
# Watch queue (refreshes every 2s)
watch kubectl get queue sc24029-user-queue -n sc24029

# Watch workloads
watch kubectl get workloads -n sc24029

# Watch pods
watch kubectl get pods -n sc24029 -l app=sllm-benchmark
```

## Continuous Monitoring Loop

```bash
NS=sc24029

while true; do
  clear
  echo "=== Queue Monitor ($(date)) ==="
  echo ""
  kubectl get queue ${NS}-user-queue -n $NS
  echo ""
  kubectl get workloads -n $NS
  echo ""
  kubectl get pods -n $NS -l app=sllm-benchmark
  echo ""
  echo "Refreshing in 5s... (Ctrl+C to stop)"
  sleep 5
done
```

## GPU Flavors on EIDF

- `gpu-a100`: Full A100 40GB GPU
- `gpu-a100-1g`: A100 MIG 1g.5gb slice
- `gpu-a100-3g`: A100 MIG 3g.20gb slice
- `gpu-a100-80`: Full A100 80GB GPU

## Useful Aliases

```bash
# Add to ~/.bashrc
alias kqueue='kubectl get queue ${NS}-user-queue -n ${NS}'
alias kwork='kubectl get workloads -n ${NS}'
alias kjobs='kubectl get jobs -n ${NS}'
alias kpods='kubectl get pods -n ${NS} -l app=sllm-benchmark'

# Then use:
export NS=sc24029
kqueue  # Check queue
kwork   # Check workloads
```
