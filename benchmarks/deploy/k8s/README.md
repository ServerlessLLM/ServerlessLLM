# Kubernetes Deployment

Deploy ServerlessLLM benchmarks to a Kubernetes cluster.

## Quick Start

```bash
# 1. Create namespace (if needed)
kubectl create namespace sllm-benchmark

# 2. Create scripts ConfigMap from local files
kubectl create configmap benchmark-scripts \
    --from-file=run-benchmark.sh=../../run-benchmark.sh \
    --from-file=download_models.py=../../download_models.py \
    --from-file=test_loading.py=../../test_loading.py \
    --from-file=benchmark_utils.py=../../benchmark_utils.py \
    --from-file=generate_report.py=../../generate_report.py \
    -n sllm-benchmark

# 3. Apply config and job
kubectl apply -f configmap.yaml -n sllm-benchmark
kubectl apply -f job.yaml -n sllm-benchmark

# 4. Monitor
kubectl logs -f job/sllm-benchmark -n sllm-benchmark
```

## Configuration

Edit `configmap.yaml` to customize:

```yaml
data:
  MODEL_NAME: "meta-llama/Meta-Llama-3-8B"  # Model to benchmark
  NUM_REPLICAS: "30"                         # Number of iterations
  MEM_POOL_SIZE: "32GB"                      # sllm-store memory pool
  BENCHMARK_TYPE: "random"                   # random or cached
```

Edit `job.yaml` to customize resources:

```yaml
resources:
  requests:
    cpu: "16"
    memory: "128Gi"
    nvidia.com/gpu: 2
```

## Gated Models (e.g., Meta-Llama)

```bash
# Create secret with HF token
kubectl create secret generic hf-token \
    --from-literal=token=$HF_TOKEN \
    -n sllm-benchmark
```

Then uncomment the HF_TOKEN env section in `job.yaml`.

## Storage Options

The default uses `emptyDir`. For better performance, edit `job.yaml`:

**hostPath** (if your cluster allows):
```yaml
- name: model-storage
  hostPath:
    path: /mnt/nvme/sllm-benchmark
    type: DirectoryOrCreate
```

**PersistentVolumeClaim**:
```yaml
- name: model-storage
  persistentVolumeClaim:
    claimName: your-pvc-name
```

## Getting Results

```bash
# Get pod name
POD=$(kubectl get pods -n sllm-benchmark -l app=sllm-benchmark -o jsonpath='{.items[0].metadata.name}')

# Copy results to local
kubectl cp sllm-benchmark/$POD:/results ./results

# View summary
cat ./results/summary.txt
```

## Cleanup

```bash
kubectl delete job sllm-benchmark -n sllm-benchmark
kubectl delete configmap benchmark-config benchmark-scripts -n sllm-benchmark
kubectl delete secret hf-token -n sllm-benchmark  # if created
```

## Cluster-Specific Notes

Different clusters may require adjustments:

- **Kueue**: Add queue label to job metadata
- **OpenShift**: May need SecurityContextConstraints
- **GKE/EKS**: Use appropriate GPU node pools
- **Resource quotas**: Adjust requests/limits to fit your quota
