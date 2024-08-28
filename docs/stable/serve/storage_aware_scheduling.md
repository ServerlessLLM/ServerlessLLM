---
sidebar_position: 0
---

# Storage Aware Scheduling

## Pre-requisites
To enable storage aware model loading scheduling, a hardware configuration file is required.
For example, the following is a sample configuration file for two servers:
```bash
echo '{
  "0": {
    "host_size": "32GB",
    "host_bandwidth": "24GB/s",
    "disk_size": "128GB",
    "disk_bandwidth": "5GB/s",
    "network_bandwidth": "10Gbps"
  },
  "1": {
    "host_size": "32GB",
    "host_bandwidth": "24GB/s",
    "disk_size": "128GB",
    "disk_bandwidth": "5GB/s",
    "network_bandwidth": "10Gbps"
  }
}' > hardware_config.json
```

We will use Docker to run a ServerlessLLM cluster in this example. Therefore, please make sure you have read the [Docker Quickstart Guide](../getting_started/docker_quickstart.md) before proceeding.

## Usage
Start a local Docker-based ray cluster.

### Step 1: Start Ray Head Node and Worker Nodes

1. Start the Ray head node.

```bash
docker run -d --name ray_head \
  --runtime nvidia \
  --network sllm \
  -p 6379:6379 \
  -p 8343:8343 \
  --gpus '"device=none"' \
  serverlessllm/sllm-serve
```

2. Start the Ray worker nodes.

Ensure that you have a directory for storing your models and set the `MODEL_FOLDER` environment variable to this directory:

```bash
export MODEL_FOLDER=path/to/models
```

```bash
docker run -d --name ray_worker_0 \
  --runtime nvidia \
  --network sllm \
  --gpus '"device=0"' \
  --env WORKER_ID=0 \
  --mount type=bind,source=$MODEL_FOLDER,target=/models \
  serverlessllm/sllm-serve-worker

docker run -d --name ray_worker_1 \
  --runtime nvidia \
  --network sllm \
  --gpus '"device=1"' \
  --env WORKER_ID=1 \
  --mount type=bind,source=$MODEL_FOLDER,target=/models \
  serverlessllm/sllm-serve-worker
```

### Step 2: Start ServerlessLLM Serve with Storage Aware Scheduler

1. Copy the hardware configuration file to the Ray head node.

```bash
docker cp hardware_config.json ray_head:/app/hardware_config.json
```

2. Start the ServerlessLLM serve with the storage aware scheduler.

```bash
docker exec ray_head sh -c "/opt/conda/bin/sllm-serve start --hardware-config /app/hardware_config.json"
```

### Step 3: Deploy Models with Placement Spec

1. Create model deployment spec files.
In this example, model "opt-2.7b" will be placed on server 0; while model "opt-1.3b" will be placed on server 1.
```bash
echo '{
    "model": "opt-2.7b",
    "backend": "transformers",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10
    },
    "placement_config": {
        "target_nodes": ["0"]
    },
    "backend_config": {
        "pretrained_model_name_or_path": "facebook/opt-2.7b",
        "device_map": "auto",
        "torch_dtype": "float16"
    }
}' > config-opt-2.7b.json
echo '{
    "model": "opt-1.3b",
    "backend": "transformers",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10
    },
    "placement_config": {
        "target_nodes": ["1"]
    },
    "backend_config": {
        "pretrained_model_name_or_path": "facebook/opt-1.3b",
        "device_map": "auto",
        "torch_dtype": "float16"
    }
}' > config-opt-1.3b.json
```

> Note: Storage aware scheduling currently only supports "transformers" backend. Support for other backends will come soon.

2. Deploy models with the placement spec files.

```bash
conda activate sllm
export LLM_SERVER_URL=http://localhost:8343/

sllm-cli deploy --config config-opt-2.7b.json
sllm-cli deploy --config config-opt-1.3b.json
```

3. Verify the deployment.

```bash
curl http://localhost:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "opt-2.7b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'

curl http://localhost:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "opt-1.3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```

As shown in the log message, the model "opt-2.7b" is scheduled on server 0, while the model "opt-1.3b" is scheduled on server 1.
```plaintext
...
(StorageAwareScheduler pid=1584) INFO 07-30 12:08:40 storage_aware_scheduler.py:138] Sorted scheduling options: [('0', 0.9877967834472656)]
(StorageAwareScheduler pid=1584) INFO 07-30 12:08:40 storage_aware_scheduler.py:145] Allocated node 0 for model opt-2.7b
...
(StorageAwareScheduler pid=1584) INFO 07-30 12:08:51 storage_aware_scheduler.py:138] Sorted scheduling options: [('1', 0.4901580810546875)]
(StorageAwareScheduler pid=1584) INFO 07-30 12:08:51 storage_aware_scheduler.py:145] Allocated node 1 for model opt-1.3b
...
```

### Step 4: Clean Up

Delete the model deployment by running the following command:

```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b
```

If you need to stop and remove the containers, you can use the following commands:

```bash
docker exec ray_head sh -c "ray stop"
docker exec ray_worker_0 sh -c "ray stop"
docker exec ray_worker_1 sh -c "ray stop"

docker stop ray_head ray_worker_0 ray_worker_1
docker rm ray_head ray_worker_0 ray_worker_1
docker network rm sllm
```