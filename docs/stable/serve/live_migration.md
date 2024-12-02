---
sidebar_position: 1
---

# Live Migration of Inference Instances

This example illustrates the live migration of inference instances in a ServerlessLLM cluster by constructing a scenario where two models are deployed to the cluster. Model `Qwen2.5-3B` is stored on both nodes, while model `Qwen2.5-1.5B` is only stored on node 0 (e.g., due to being less popular). This example will show a locality-contention scenario where `Qwen2.5-3B` is being served on node 0 but `Qwen2.5-1.5B` is requested to be served on the same node for optimal locality. We will find that:

- **Without migration**, `Qwen2.5-1.5B` would have to wait for the completion of the ongoing inference instance of `Qwen2.5-3B` on node 0.
- **With live migration**, the ongoing inference instance of `Qwen2.5-3B` is migrated to node 1, and `Qwen2.5-1.5B` is allocated to node 0, thus can be served immediately.

## Prerequisites

To run this example, we will use Docker Compose to set up a ServerlessLLM cluster. Before proceeding, please ensure you have read the [Docker Quickstart Guide](../getting_started/docker_quickstart.md).

**Requirements:**

- **Two GPUs** are required to illustrate the live migration of inference instances.
- **At least 20 GB of host memory** (this can be adjusted by using smaller models).
- **ServerlessLLM version 0.6**: Ensure you have `sllm==0.6` and `sllm-store==0.6` installed.

##

Start a local Docker-based ray cluster using Docker Compose.

### Clone the ServerlessLLM Repository

If you haven't already, clone the ServerlessLLM repository:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/examples/live_migration
```

### Configure the Model Directory

Create a directory on your host machine where models will be stored, and set the MODEL_FOLDER environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

The Docker Compose configuration is already located in the `examples/live_migration` directory.

## Test ServerlessLLM Without Live Migration

1. **Start the ServerlessLLM Services Using Docker Compose**

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

2. **Deploy Models with the Placement Spec Files**

Activate the ServerlessLLM environment and set the server URL:
```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343/
```

Deploy the models:
```bash
sllm-cli deploy --config config-qwen-1.5b.json
sllm-cli deploy --config config-qwen-3b.json
```

3. **Verify the Deployment**

Start two inference requests in parallel. The first request is for `Qwen2.5-3B`, and the second request, sent shortly after, is for `Qwen2.5-1.5B`. The `sleep` command is used to introduce a short interval between the two requests:

```bash
curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Could you share a story of the history of Computer Science?"}
        ],
        "max_tokens": 1024
    }' &

sleep 3

curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ],
        "max_tokens": 64
    }'
```

Since `Qwen2.5-3B` is requested first, `Qwen2.5-1.5B` must wait for the ongoing inference instance of `Qwen2.5-3B` to complete on node 0 before it can start processing.


4. Clean up.

```bash
docker compose down
```

## Test ServerlessLLM With Live Migration

1. **Start the ServerlessLLM Services with Live Migration Enabled**

Use the following command to start the ServerlessLLM services with live migration enabled. This configuration includes the `enable-migration.yml` file:

```bash
docker compose -f docker-compose.yml -f enable-migration.yml up -d --build
```

This command will start the Ray head node and two worker nodes, enabling the live migration feature.

2. **Deploy Models with the Placement Spec Files**

Activate the ServerlessLLM environment and set the server URL:

```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343/
```

Deploy the models:

```bash
sllm-cli deploy --config config-qwen-1.5b.json
sllm-cli deploy --config config-qwen-3b.json
```

3. **Verify the Deployment**

Start two inference requests in parallel. The first request is for `Qwen2.5-3B`, and the second request, sent shortly after, is for `Qwen2.5-1.5B`. The `sleep` command is used to introduce a short interval between the two requests:

```bash
curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Could you share a story of the history of Computer Science?"}
        ],
        "max_tokens": 1024
    }' &

sleep 3

curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ],
        "max_tokens": 64
    }'
```

According to the response, you should observe that `Qwen2.5-1.5B` completes ahead of `Qwen2.5-3B`. This is because the ongoing inference instance of `Qwen2.5-3B` is live-migrated from node 0 to node 1, allowing `Qwen2.5-1.5B` to be served immediately on node 0.

As shown in the log message, the ongoing inference instance of the model `Qwen/Qwen2.5-3B-Instruct` is live-migrated from node 0 to node 1. And model `Qwen/Qwen2.5-1.5B-Instruct` is allocated to node 0.

```log
TODO
```

4. Clean up.

```bash
docker compose down
```


