# Running Benchmarks on EIDF Cluster

EIDF has specific requirements that differ from standard Kubernetes:

## Key Differences

1. **`generateName` required**: Jobs must use `generateName` instead of `name`
2. **No `hostPath` volumes**: Security policy blocks direct host access
3. **Namespace required**: Must specify namespace explicitly

## Quick Setup

### 1. Find Your Namespace
```bash
kubectl get namespaces
# Or check where configmap was created:
kubectl get configmap -A | grep benchmark
```

### 2. Update Configuration Files

Edit these files and replace `<YOUR_NAMESPACE>` with your actual namespace (e.g., `sc24029`):

- `k8s/benchmark-configmap.yaml`
- `k8s/benchmark-scripts-configmap.yaml`
- `k8s/benchmark-job-eidf.yaml`

### 3. Fill Resource Placeholders

In `k8s/benchmark-job-eidf.yaml`, replace:
- `<CPU_REQUEST>` / `<CPU_LIMIT>`: e.g., `"8"`
- `<MEMORY_REQUEST>` / `<MEMORY_LIMIT>`: e.g., `"32Gi"`
- `<GPU_COUNT>`: e.g., `1`

### 4. Deploy

```bash
# Deploy config
kubectl apply -f k8s/benchmark-configmap.yaml
kubectl apply -f k8s/benchmark-scripts-configmap.yaml

# Run benchmark (uses generateName, creates unique job name)
kubectl create -f k8s/benchmark-job-eidf.yaml

# Find the job (name will be sllm-benchmark-xxxxx)
kubectl get jobs

# View logs
kubectl logs -f job/sllm-benchmark-xxxxx
```

## Storage

EIDF doesn't allow `hostPath`. The job uses `emptyDir` by default:

- **Model storage**: `emptyDir` with `sizeLimit: 100Gi`
- **Results**: `emptyDir` (temporary)

### To Persist Results

Create a PVC first:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: benchmark-results
  namespace: <YOUR_NAMESPACE>
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 50Gi
```

Then update the job to use it:
```yaml
volumes:
- name: results
  persistentVolumeClaim:
    claimName: benchmark-results
```

## Scripts Download

The benchmark downloads scripts from GitHub at runtime:
- Branch: `claude/serverlessllm-docker-setup-01SxEhpoNSDpRQp1iKqWfyRS` (default)
- Override with env var: `BRANCH=main`

## Troubleshooting

**"generateName should be used"**
- Use `kubectl create` instead of `kubectl apply`
- Or the file already uses `generateName`

**"hostPath volumes" warning**
- Use `emptyDir` or PVC
- Already fixed in `benchmark-job-eidf.yaml`

**Job not found**
- Job names are auto-generated: `sllm-benchmark-xxxxx`
- Use: `kubectl get jobs | grep sllm-benchmark`

**Results lost after job completes**
- `emptyDir` is temporary
- Use PVC for persistent results
- Or copy results: `kubectl cp <pod>:/results ./local-results`

## Example: Complete Workflow

```bash
# 1. Set your namespace
export NS=sc24029

# 2. Update all files (sed example for Linux/Mac)
sed -i "s/<YOUR_NAMESPACE>/$NS/g" k8s/benchmark-*.yaml

# 3. Update resources in benchmark-job-eidf.yaml
# Edit manually or use sed

# 4. Deploy
kubectl apply -f k8s/benchmark-configmap.yaml -n $NS
kubectl apply -f k8s/benchmark-scripts-configmap.yaml -n $NS
kubectl create -f k8s/benchmark-job-eidf.yaml -n $NS

# 5. Monitor
JOB=$(kubectl get jobs -n $NS | grep sllm-benchmark | awk '{print $1}')
kubectl logs -f job/$JOB -n $NS

# 6. Get results
POD=$(kubectl get pods -n $NS | grep sllm-benchmark | awk '{print $1}')
kubectl cp $NS/$POD:/results ./results
```
