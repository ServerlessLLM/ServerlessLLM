# NVMe Local Storage Setup for sllm-store on EIDF

This guide explains how to use local NVMe SSD storage with auto-detection for sllm-store on the EIDF Kubernetes cluster.

## Overview

The setup includes:
1. **Auto-detection init container** - Scans for NVMe devices and validates storage
2. **Debug logging** - Comprehensive logs for troubleshooting
3. **hostPath volumes** - Direct access to local NVMe storage
4. **Performance validation** - Tests write performance before starting sllm-store

## Prerequisites

1. Worker nodes must have NVMe SSDs installed
2. NVMe devices must be mounted to a path (e.g., `/mnt/nvme`, `/data/nvme`)
3. You need to know the mount path on the worker nodes

## Step 1: Find NVMe Mount Path on Your Cluster

First, you need to determine where NVMe drives are mounted on EIDF worker nodes.

### Option A: SSH to a worker node (if you have access)
```bash
# List all NVMe devices
lsblk | grep nvme

# Show mount points
df -h | grep nvme

# Common mount locations:
# - /mnt/nvme
# - /data
# - /scratch
# - /local-ssd
```

### Option B: Run a detection pod
```bash
kubectl run nvme-finder --image=ubuntu:20.04 --restart=Never -it --rm \
  --overrides='{"spec":{"hostNetwork":true,"volumes":[{"name":"dev","hostPath":{"path":"/dev"}},{"name":"sys","hostPath":{"path":"/sys"}}],"containers":[{"name":"nvme-finder","image":"ubuntu:20.04","command":["/bin/bash"],"stdin":true,"tty":true,"volumeMounts":[{"name":"dev","mountPath":"/dev"},{"name":"sys","mountPath":"/sys"}],"securityContext":{"privileged":true}}]}}' \
  -- bash -c "apt update && apt install -y nvme-cli pciutils && lsblk && df -h && nvme list"
```

### Option C: Ask EIDF support
Contact EIDF support to ask:
- "Where are NVMe SSDs mounted on GPU worker nodes?"
- Typical paths they might provide: `/mnt/nvme`, `/data`, `/scratch`

## Step 2: Deploy the ConfigMap

The ConfigMap contains the NVMe detection script:

```bash
# Edit the namespace in nvme-detect-configmap.yaml
# Replace <YOUR_NAMESPACE> with your namespace

kubectl apply -f k8s/nvme-detect-configmap.yaml
```

Verify:
```bash
kubectl get configmap nvme-detect-script
```

## Step 3: Configure the Job Manifest

Edit `k8s/sllm-store-job-nvme.yaml` and fill in the placeholders:

### Required placeholders:
```yaml
metadata:
  labels:
    kueue.x-k8s.io/queue-name: <YOUR_NAMESPACE>-user-queue  # e.g., "my-project-user-queue"

containers:
- name: sllm-store
  image: <YOUR_REGISTRY>/sllm-store:latest  # e.g., "docker.io/username/sllm-store:latest"

  resources:
    requests:
      cpu: "<CPU_REQUEST>"        # e.g., "8"
      memory: "<MEMORY_REQUEST>"  # e.g., "32Gi"
      nvidia.com/gpu: <GPU_COUNT> # e.g., 1
    limits:
      cpu: "<CPU_LIMIT>"          # e.g., "8"
      memory: "<MEMORY_LIMIT>"    # e.g., "32Gi"
      nvidia.com/gpu: <GPU_COUNT> # e.g., 1

  args:
  - |
    ...
    exec sllm-store start --storage-path /models --mem-pool-size <MEM_POOL_SIZE>
    # e.g., --mem-pool-size 20GB

volumes:
- name: model-storage
  hostPath:
    path: <NVME_MOUNT_PATH>/sllm-models  # e.g., "/mnt/nvme/sllm-models"
```

### Example with values filled in:
```yaml
volumes:
- name: model-storage
  hostPath:
    path: /mnt/nvme/sllm-models  # Assuming NVMe is mounted at /mnt/nvme
    type: DirectoryOrCreate
```

## Step 4: Deploy sllm-store

```bash
# Deploy the job
kubectl apply -f k8s/sllm-store-job-nvme.yaml

# Check job status
kubectl get jobs

# Get pod name
POD_NAME=$(kubectl get pods -l app=sllm-store -o jsonpath='{.items[0].metadata.name}')

# View init container logs (NVMe detection)
kubectl logs $POD_NAME -c nvme-detect

# View main container logs
kubectl logs $POD_NAME -c sllm-store -f
```

## Step 5: Review Detection Logs

The init container will output detailed information about:
- All NVMe devices found
- Device sizes, models, serial numbers
- Mount points and available space
- Filesystem type
- I/O scheduler settings
- Write performance test results

### Example output:
```
=== NVMe SSD Detection Script ===
Timestamp: 2024-01-15 10:30:00
Hostname: worker-node-01

--- NVMe Block Devices ---
Found NVMe devices:
/dev/nvme0n1

--- NVMe Device Details ---
Device: /dev/nvme0n1
  Size: 3725.29 GB
  Model: Samsung SSD 970 EVO Plus 4TB
  Serial: S4P2NF0R123456
  Mount point: /mnt/nvme
  Available space: 3.2T

--- Checking Model Storage Path ---
Model storage path: /models
  Path exists: YES
  Writable: YES
  Space available: 3.2T
  Filesystem type: ext4
  Mounted from device: /dev/nvme0n1
  Storage type: NVMe SSD ✓
  I/O Scheduler: none

  Testing write performance...
  100+0 records in
  100+0 records out
  104857600 bytes (105 MB, 100 MiB) copied, 0.123456 s, 849 MB/s
  Write test: PASSED

=== Detection Complete ===
```

## Step 6: Access Debug Logs from the Running Pod

The detection log is saved to `/debug/nvme-detection.log` and is accessible from the main container:

```bash
# View the detection log
kubectl exec $POD_NAME -c sllm-store -- cat /debug/nvme-detection.log

# Copy the log to your local machine
kubectl cp $POD_NAME:/debug/nvme-detection.log ./nvme-detection.log -c sllm-store
```

## Troubleshooting

### Init container fails: "No NVMe devices found"

**Possible causes:**
1. Worker node doesn't have NVMe SSDs
2. NVMe driver not loaded
3. Insufficient permissions

**Solutions:**
- Check node selector to ensure pod runs on nodes with NVMe
- Label nodes with NVMe: `kubectl label nodes <node-name> storage.type=nvme`
- Add nodeSelector to job manifest:
  ```yaml
  nodeSelector:
    storage.type: nvme
  ```

### Path does not exist error

**Possible causes:**
- Incorrect mount path specified
- NVMe not mounted on the node

**Solutions:**
- Verify mount path using the detection pod (Step 1)
- Check with `kubectl get nodes -o wide` and SSH to node
- Use `DirectoryOrCreate` type to auto-create directories

### Permission denied errors

**Possible causes:**
- Directory permissions don't allow pod user to write

**Solutions:**
- Uncomment `CHOWN_USER` in init container env vars:
  ```yaml
  env:
  - name: CHOWN_USER
    value: "1000"  # Or the UID your container runs as
  ```

### Low write performance in test

**Possible causes:**
- Not actually using NVMe (using slow storage)
- I/O scheduler misconfigured
- Disk nearly full

**Solutions:**
- Verify "Storage type: NVMe SSD ✓" in logs
- Check I/O scheduler (should be `none` or `mq-deadline` for NVMe)
- Ensure sufficient free space

### Pod doesn't start: "forbidden" or quota errors

**Possible causes:**
- Resource quota exceeded
- Queue configuration issues

**Solutions:**
- Check quotas: `kubectl describe resourcequota`
- Verify queue name matches your namespace
- Reduce resource requests

## Advanced Configuration

### Using Specific NVMe Devices

If you have multiple NVMe devices and want to use a specific one:

1. Note the device from detection logs (e.g., `/dev/nvme1n1`)
2. Mount it to a specific path on the host
3. Update the hostPath in your manifest

### Performance Tuning

For optimal performance, ensure:
- NVMe I/O scheduler is set to `none` or `mq-deadline`
- Filesystem is ext4 or xfs (not NFS)
- Sufficient free space (fragmentation affects performance)

### Node Affinity

To ensure pods only run on nodes with NVMe:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: storage.type
          operator: In
          values:
          - nvme
```

## Monitoring

### Check storage usage:
```bash
kubectl exec $POD_NAME -c sllm-store -- df -h /models
```

### Watch logs in real-time:
```bash
kubectl logs -f $POD_NAME -c sllm-store
```

### Get storage performance stats:
```bash
kubectl exec $POD_NAME -c sllm-store -- bash -c "dd if=/dev/zero of=/models/test bs=1G count=1 oflag=direct && rm /models/test"
```

## Summary

The NVMe auto-detection setup:
1. ✅ Automatically detects NVMe devices
2. ✅ Validates storage is actually NVMe
3. ✅ Tests write performance
4. ✅ Logs all information for debugging
5. ✅ Fails early if storage issues detected

All you need to provide is the NVMe mount path in `<NVME_MOUNT_PATH>`.
