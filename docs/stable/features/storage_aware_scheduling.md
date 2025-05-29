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
cd ServerlessLLM/examples/storage_aware_scheduling
```

### Step 2: Configuration

Set the Model Directory. Create a directory on your host machine where models will be stored and set the `MODEL_FOLDER` environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

### Step 3: Enable Storage Aware Scheduling in Docker Compose

The Docker Compose configuration is already located in the `examples/storage_aware_scheduling` directory. To activate storage-aware scheduling, ensure the `docker-compose.yml` file includes the necessary configurations(`sllm_head` service should include the `--enable-storage-aware` command).

:::tip
Recommend to adjust the number of GPUs and `mem_pool_size` based on the resources available on your machine.
:::


### Step 4: Start the Services

Start the ServerlessLLM services using Docker Compose:

```bash
docker compose up -d
```

This command will start the Ray head node and two worker nodes defined in the `docker-compose.yml` file.

:::tip
Use the following command to monitor the logs of the head node:

```bash
docker logs -f sllm_head
```
:::

### Step 5: Deploy Models with Placement Spec

In the `examples/storage_aware_scheduling` directory, the example configuration files (`config-opt-2.7b.json` and `config-opt-1.3b.json`) are already given.

> Note: Storage aware scheduling currently only supports the "transformers" backend. Support for other backends will come soon.

2. Deploy models with the placement spec files.

```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343

sllm-cli deploy --config config-opt-2.7b.json
sllm-cli deploy --config config-opt-1.3b.json
```

3. Verify the deployment.

```bash
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-2.7b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'

curl $LLM_SERVER_URL/v1/chat/completions \
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

