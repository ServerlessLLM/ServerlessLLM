# Quick Deploy to EIDF

## One-Command Deploy

```bash
# Deploy
NS=sc24029 k8s/deploy-benchmark.sh

# Monitor progress
NS=sc24029 k8s/monitor-benchmark.sh
```

Done! Scripts auto-replace all `<PLACEHOLDERS>` from environment variables.

## Configuration

Set environment variables before running:

```bash
# Required
export NS=sc24029              # Your namespace

# Optional (with defaults)
export CPU=8                   # CPU cores (default: 8)
export MEMORY=32Gi            # Memory (default: 32Gi)
export GPU=1                   # GPU count (default: 1)
export NVME_PATH=/nvme        # NVMe mount path (default: /nvme)
export MODEL_NAME=facebook/opt-1.3b
export NUM_REPLICAS=10

# Deploy
k8s/deploy-benchmark.sh
```

## What It Does

1. ✅ Validates namespace is set
2. ✅ Processes all YAML files with your values
3. ✅ Applies ConfigMaps
4. ✅ Creates benchmark job
5. ✅ Shows commands to monitor progress

## Example Output

```
=== EIDF Benchmark Deployment ===
Namespace: sc24029
Resources: 8 CPU, 32Gi RAM, 1 GPU
NVMe Path: /nvme
Model: facebook/opt-1.3b (10 replicas)

Processing manifests...
1. Processing benchmark-configmap.yaml...
2. Processing benchmark-scripts-configmap.yaml...
3. Processing benchmark-job-eidf.yaml...

Applying configurations...
  - benchmark-configmap
  - benchmark-scripts

Creating benchmark job...

=== Deployment Complete ===

Job created: sllm-benchmark-abc123

Monitor logs:
  kubectl logs -f job/sllm-benchmark-abc123 -n sc24029

Check status:
  kubectl get job/sllm-benchmark-abc123 -n sc24029

Get results (after completion):
  kubectl cp sc24029/sllm-benchmark-abc123-xyz:/results ./results
```

## NVMe Storage

The script uses `hostPath` to mount NVMe at `$NVME_PATH/sllm-benchmark-models`.

**Note**: You'll see a warning about `hostPath volumes` - this is normal and non-blocking on EIDF. The job will still run.

If hostPath is blocked:
- Edit `benchmark-job-eidf.yaml`
- Uncomment `emptyDir` section
- Comment out `hostPath` section

## Find Your NVMe Path

```bash
# On the worker node or via detection
kubectl run nvme-check --image=ubuntu:20.04 --rm -it --restart=Never \
  --overrides='{"spec":{"volumes":[{"name":"root","hostPath":{"path":"/"}}],"containers":[{"name":"nvme-check","image":"ubuntu:20.04","command":["/bin/sh","-c","df -h | grep nvme"],"stdin":true,"tty":true,"volumeMounts":[{"name":"root","mountPath":"/host"}],"securityContext":{"privileged":true}}]}}'
```

Common paths:
- `/nvme`
- `/mnt/nvme`
- `/data`
- `/scratch`

## Cleanup

```bash
# Delete completed jobs
kubectl delete jobs -n $NS -l app=sllm-benchmark

# Delete all resources
kubectl delete -n $NS \
  configmap/benchmark-config \
  configmap/benchmark-scripts-full \
  jobs -l app=sllm-benchmark
```

## Troubleshooting

**"NS environment variable not set"**
```bash
NS=your-namespace k8s/deploy-benchmark.sh
```

**"hostPath volumes" warning appears but job runs**
- This is expected - the warning is non-blocking

**Job fails with storage errors**
- Check NVMe path: `NVME_PATH=/correct/path k8s/deploy-benchmark.sh`
- Or use emptyDir (edit yaml manually)

**"command not found: envsubst"**
- Script auto-falls back to `sed` - no action needed
