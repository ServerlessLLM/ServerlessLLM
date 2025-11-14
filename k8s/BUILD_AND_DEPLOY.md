# Building and Deploying sllm-store on EIDF Kubernetes Cluster

This guide shows how to build and deploy sllm-store using the `Dockerfile.k8s` on the EIDF GPU cluster.

**For NVMe local storage setup with auto-detection, see [NVME_SETUP.md](NVME_SETUP.md)**

## Prerequisites

1. Access to EIDF GPU cluster with `kubectl` configured
2. Docker installed locally or access to a machine where you can build images
3. A container registry accessible from the EIDF cluster (Docker Hub, GitHub Container Registry, etc.)
4. Models stored in accessible storage (PVC, hostPath, or NFS)

## Step 1: Build the Docker Image

From the `sllm_store` directory:

```bash
cd /path/to/ServerlessLLM/sllm_store

# Build the image
docker build -f Dockerfile.k8s -t <YOUR_REGISTRY>/sllm-store:latest .

# Example with Docker Hub:
# docker build -f Dockerfile.k8s -t yourusername/sllm-store:latest .

# Example with GitHub Container Registry:
# docker build -f Dockerfile.k8s -t ghcr.io/yourusername/sllm-store:latest .
```

## Step 2: Push the Image to Registry

```bash
# Login to your registry (if needed)
docker login <YOUR_REGISTRY>

# Push the image
docker push <YOUR_REGISTRY>/sllm-store:latest
```

## Step 3: Prepare Model Storage

Choose one of the storage options and configure accordingly:

### Option A: Using PersistentVolumeClaim (Recommended)

1. Edit `k8s/storage-pvc.yaml` and fill in the placeholders
2. Create the PVC:
   ```bash
   kubectl apply -f k8s/storage-pvc.yaml
   ```
3. Verify PVC is bound:
   ```bash
   kubectl get pvc <YOUR_PVC_NAME>
   ```

### Option B: Using hostPath

Ensure your models are located on the worker node at a specific path (e.g., `/data/models`).
Update the volume configuration in `sllm-store-job.yaml` to use hostPath (see comments in file).

### Option C: Using emptyDir

For testing only - data will be lost when pod terminates.
Update the volume configuration in `sllm-store-job.yaml` to use emptyDir (see comments in file).

## Step 4: Configure the Job Manifest

Edit `k8s/sllm-store-job.yaml` and replace all placeholders:

### Required placeholders:
- `<YOUR_NAMESPACE>`: Your Kubernetes namespace (e.g., `my-project`)
- `<YOUR_REGISTRY>`: Your container registry (e.g., `docker.io/yourusername`)
- `<CPU_REQUEST>`: CPU request (e.g., `4` or `4000m`)
- `<CPU_LIMIT>`: CPU limit (e.g., `4` or `4000m`)
- `<MEMORY_REQUEST>`: Memory request (e.g., `16Gi`)
- `<MEMORY_LIMIT>`: Memory limit (e.g., `16Gi`)
- `<GPU_COUNT>`: Number of GPUs (e.g., `1`)
- `<MEM_POOL_SIZE>`: Memory pool size for sllm-store (e.g., `4GB`, `8GB`)
- `<YOUR_PVC_NAME>`: If using PVC, your PVC name

### Determining mem-pool-size:
The `--mem-pool-size` should be **larger than your largest model size**. For example:
- For 7B models (~14GB in FP16): use `16GB` or `20GB`
- For 13B models (~26GB in FP16): use `32GB`
- For 70B models (~140GB in FP16): use `160GB`

### Optional configurations:
- Uncomment `nodeSelector` to specify GPU type
- Uncomment `tolerations` if your cluster uses GPU node taints
- Adjust `args` to add more sllm-store options

## Step 5: Deploy to Kubernetes

```bash
# Apply the job
kubectl apply -f k8s/sllm-store-job.yaml

# Check job status
kubectl get jobs

# Check pod status
kubectl get pods -l app=sllm-store

# View logs
kubectl logs -f job/sllm-store-worker

# If the pod name is known:
# kubectl logs -f <pod-name>
```

## Step 6: Verify Deployment

Check the logs for successful startup:

```bash
kubectl logs -f job/sllm-store-worker
```

You should see output indicating that sllm-store has started successfully, such as:
```
Starting checkpoint store server...
Server listening on port 8073
```

## Step 7: Access sllm-store

### From within the cluster:
If you need to access sllm-store from other pods, create a Service:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: sllm-store-service
spec:
  selector:
    app: sllm-store
  ports:
  - protocol: TCP
    port: 8073
    targetPort: 8073
```

Then access via `sllm-store-service:8073` from other pods.

### From outside the cluster:
Use port-forwarding:
```bash
kubectl port-forward job/sllm-store-worker 8073:8073
```

## Troubleshooting

### Pod not starting:
```bash
kubectl describe pod <pod-name>
```

### Check resource availability:
```bash
kubectl describe nodes
```

### View detailed logs:
```bash
kubectl logs <pod-name> --previous  # View logs from previous run if crashed
```

### Common issues:

1. **Image pull errors**: Verify registry credentials and image name
2. **Resource quota exceeded**: Check namespace resource quotas
3. **PVC not binding**: Verify storage class and capacity
4. **GPU not allocated**: Check GPU availability on nodes

## Cleaning Up

```bash
# Delete the job
kubectl delete -f k8s/sllm-store-job.yaml

# Delete the PVC (if needed)
kubectl delete -f k8s/storage-pvc.yaml
```

## Example: Complete Configuration

Here's an example with all placeholders filled:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: sllm-store-worker
  labels:
    kueue.x-k8s.io/queue-name: my-project-user-queue
spec:
  template:
    metadata:
      labels:
        app: sllm-store
    spec:
      restartPolicy: Never
      containers:
      - name: sllm-store
        image: docker.io/myusername/sllm-store:latest
        resources:
          requests:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: 1
          limits:
            cpu: "4"
            memory: "16Gi"
            nvidia.com/gpu: 1
        command: ["sllm-store", "start"]
        args:
        - "--storage-path"
        - "/models"
        - "--mem-pool-size"
        - "20GB"
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: sllm-models-pvc
      nodeSelector:
        nvidia.com/gpu.product: 'NVIDIA-A100-SXM4-40GB'
```
