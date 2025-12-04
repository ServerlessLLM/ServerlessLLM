# Quick Deploy to EIDF

## One-Command Deploy

```bash
# Deploy with CLI flags (recommended)
k8s/deploy-benchmark.sh --namespace sc24029

# Monitor progress (includes queue status)
k8s/monitor-benchmark.sh --namespace sc24029

# Check queue status only
k8s/monitor-queue.sh --namespace sc24029
```

Done! The deploy script auto-replaces all `<PLACEHOLDERS>` in YAML files.

## Configuration

Use command-line flags for easy configuration:

```bash
# Basic deployment
k8s/deploy-benchmark.sh --namespace sc24029

# Gated models (e.g., Meta-Llama) - provide HF token
k8s/deploy-benchmark.sh \
    --namespace sc24029 \
    --hf-token $HF_TOKEN \
    --model-name meta-llama/Meta-Llama-3-8B

# Use existing K8s secret for HF token
k8s/deploy-benchmark.sh \
    --namespace sc24029 \
    --hf-secret my-hf-secret \
    --model-name meta-llama/Meta-Llama-3-8B

# Full customization with CLI flags
k8s/deploy-benchmark.sh \
    --namespace sc24029 \
    --cpu 16 \
    --memory 256Gi \
    --gpu 2 \
    --nvme-path /nvme \
    --model-name meta-llama/Meta-Llama-3-8B \
    --num-replicas 50

# See all available options
k8s/deploy-benchmark.sh --help
```

**Backward compatibility:** Environment variables are still supported (e.g., `NS=sc24029 k8s/deploy-benchmark.sh`), but CLI flags are recommended.

## What It Does

1. ✅ Validates namespace is set
2. ✅ Processes all YAML files with your values
3. ✅ Creates HF token secret (if `--hf-token` provided)
4. ✅ Applies ConfigMaps
5. ✅ Creates benchmark job
6. ✅ Shows commands to monitor progress

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

**"Namespace not specified"**
```bash
k8s/deploy-benchmark.sh --namespace your-namespace
```

**"hostPath volumes" warning appears but job runs**
- This is expected - the warning is non-blocking

**Job fails with storage errors**
- Check NVMe path: `k8s/deploy-benchmark.sh --nvme-path /correct/path`
- Or use emptyDir (edit yaml manually)

**"command not found: envsubst"**
- Script auto-falls back to `sed` - no action needed

**Gated model access denied (401/403 error)**
```bash
# Provide HF token when deploying
k8s/deploy-benchmark.sh --namespace sc24029 --hf-token $HF_TOKEN

# Or create secret manually
kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN -n sc24029
```
