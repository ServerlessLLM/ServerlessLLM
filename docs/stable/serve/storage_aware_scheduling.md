---
sidebar_position: 0
---

# Storage Aware Scheduling with Docker Compose

## Pre-requisites

We will use Docker Compose to run a ServerlessLLM cluster in this example. Therefore, please make sure you have read the [Docker Quickstart Guide](../getting_started/docker_quickstart.md) before proceeding.

## Usage

Start a local Docker-based ray cluster using Docker Compose.

### Step 1: Clone the ServerlessLLM Repository

If you haven't already, clone the ServerlessLLM repository:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd serverlessllm/examples/docker/
```

### Step 2: Configuration

Set the Model Directory. Create a directory on your host machine where models will be stored and set the `MODEL_FOLDER` environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

### Step 3: Enable Storage Aware Scheduling in Docker Compose

To activate storage-aware scheduling, edit the `docker-compose.yml` file to enable the feature. Update the `sllm_head` service to include the `--enable_storage_aware` command, and adjust the configuration for the worker nodes. The following is an example of the updated `docker-compose.yml` file:

```yaml
services:
  # Head Node
  sllm_head:
    build:
      context: ../../
      dockerfile: Dockerfile  # Ensure this points to your head node Dockerfile
    image: serverlessllm/sllm-serve
    container_name: sllm_head
    environment:
      - MODEL_FOLDER=${MODEL_FOLDER}
    ports:
      - "6379:6379"    # Redis port
      - "8343:8343"    # ServerlessLLM port
    networks:
      - sllm_network
    command: ["--enable_storage_aware"]  # Enable storage-aware scheduling

  # Worker Node 0
  sllm_worker_0:
    build:
      context: ../../
      dockerfile: Dockerfile.worker  # Ensure this points to your worker Dockerfile
    image: serverlessllm/sllm-serve-worker
    container_name: sllm_worker_0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              # count: 1  # Assigns 1 GPU to the worker
              device_ids: ["0"]
    environment:
      - WORKER_ID=0
      - STORAGE_PATH=/models
    networks:
      - sllm_network
    volumes:
      - ${MODEL_FOLDER}:/models
    command: ["-mem_pool_size", "32", "-registration_required", "true"]

  # Worker Node 1
  sllm_worker_1:
    build:
      context: ../../
      dockerfile: Dockerfile.worker
    image: serverlessllm/sllm-serve-worker
    container_name: sllm_worker_1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              # count: 1  # Assigns 1 GPU to the worker
              device_ids: ["1"]
    environment:
      - WORKER_ID=1
      - MODEL_FOLDER=${MODEL_FOLDER}
    networks:
      - sllm_network
    volumes:
      - ${MODEL_FOLDER}:/models
    command: ["-mem_pool_size", "32", "-registration_required", "true"]

networks:
  sllm_network:
    driver: bridge
    name: sllm

```

:::tip
Recommend to adjust the number of GPUs and `mem_pool_size` based on the resources available on your machine.
:::


### Step 4: Start the Services

Start the ServerlessLLM services using Docker Compose:

```bash
docker compose up -d --build
```

This command will start the Ray head node and two worker nodes defined in the `docker-compose.yml` file.

:::tip
Use the following command to monitor the logs of the head node:

```bash
docker logs -f sllm_head
```
:::

### Step 5: Deploy Models with Placement Spec

1. Create model deployment spec files. In this example, model "facebook/opt-2.7b" will be placed on server 0, while model "facebook/opt-1.3b" will be placed on server 1.

```bash
echo '{
    "model": "facebook/opt-2.7b",
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
    "model": "facebook/opt-1.3b",
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

> Note: Storage aware scheduling currently only supports the "transformers" backend. Support for other backends will come soon.

2. Deploy models with the placement spec files.

```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343/

sllm-cli deploy --config config-opt-2.7b.json
sllm-cli deploy --config config-opt-1.3b.json
```

3. Verify the deployment.

```bash
curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-2.7b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'

curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-1.3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```

As shown in the log message, the model "facebook/opt-2.7b" is scheduled on server 0, while the model "facebook/opt-1.3b" is scheduled on server 1.

```log
(StorageAwareScheduler pid=1543) INFO 11-12 23:48:27 storage_aware_scheduler.py:137] Sorted scheduling options: [('0', 4.583079601378258)]
(StorageAwareScheduler pid=1543) INFO 11-12 23:48:27 storage_aware_scheduler.py:144] Allocated node 0 for model facebook/opt-2.7b
(StorageAwareScheduler pid=1543) INFO 11-12 23:48:38 storage_aware_scheduler.py:137] Sorted scheduling options: [('1', 2.266678696047572)]
(StorageAwareScheduler pid=1543) INFO 11-12 23:48:38 storage_aware_scheduler.py:144] Allocated node 1 for model facebook/opt-1.3b
```

### Step 6: Clean Up

Delete the model deployment by running the following command:

```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b
```

If you need to stop and remove the containers, you can use the following commands:

```bash
docker compose down
```

