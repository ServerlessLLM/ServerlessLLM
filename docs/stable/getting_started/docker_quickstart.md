---
sidebar_position: 2
---

# Docker Quickstart Guide

This guide will help you get started with the basics of using ServerlessLLM with Docker. Please make sure you have Docker installed on your system and have installed ServerlessLLM CLI following the [installation guide](./installation.md).

## Pre-requirements

Ensure you have the following pre-requirements installed:

1. **GPUs**: Ensure you have at least 2 GPUs available. If more GPUs are provided, you can adjust the number of workers and the number of devices assigned to each worker.
2. **NVIDIA Docker Toolkit**: This allows Docker to use NVIDIA GPUs. You can find the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Local Test Using Docker

First, let's start a local Docker-based ray cluster to test the ServerlessLLM.

### Step 1: Build Docker Images

Run the following commands to build the Docker images:

```bash
docker build . -t serverlessllm/sllm-serve
docker build -f Dockerfile.worker . -t serverlessllm/sllm-serve-worker
```

### Step 2: Configuration

Ensure that you have a directory for storing your models and set the `MODEL_FOLDER` environment variable to this directory:

```bash
export MODEL_FOLDER=path/to/models
```

Also, check if the Docker network `sllm` exists and create it if it doesn't:

```bash
if ! docker network ls | grep -q "sllm"; then
  echo "Docker network 'sllm' does not exist. Creating network..."
  docker network create sllm
else
  echo "Docker network 'sllm' already exists."
fi
```

### Step 3: Start the Ray Head and Worker Nodes

Run the following commands to start the Ray head node and worker nodes:

#### Start Ray Head Node

```bash
docker run -d --name ray_head \
  --runtime nvidia \
  --network sllm \
  -p 6379:6379 \
  -p 8343:8343 \
  --gpus '"device=none"' \
  serverlessllm/sllm-serve
```

#### Start Ray Worker Nodes

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

### Step 4: Start ServerlessLLM Serve

Run the following command to start the ServerlessLLM serve:

```bash
docker exec ray_head sh -c "/opt/conda/bin/sllm-serve start"
```

### Step 5: Deploy a Model Using sllm-cli

Open a new terminal, activate the `sllm` environment, and set the `LLM_SERVER_URL` environment variable:

```bash
conda activate sllm
export LLM_SERVER_URL=http://localhost:8343/
```

Deploy a model to the ServerlessLLM server using the `sllm-cli`:

```bash
sllm-cli deploy --model facebook/opt-1.3b
```
> Note: This command will spend some time downloading the model from the Hugging Face Model Hub.
> You can use any model from the [Hugging Face Model Hub](https://huggingface.co/models) by specifying the model name in the `--model` argument.

Expected output:

```plaintext
INFO xx-xx xx:xx:xx deploy.py:36] Deploying model facebook/opt-1.3b with default configuration.
INFO xx-xx xx:xx:xx deploy.py:49] Model registered successfully.
```

### Step 6: Query the Model

Now, you can query the model by any OpenAI API client. For example, you can use the following Python code to query the model:
```bash
curl http://localhost:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-1.3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```

Expected output:

```plaintext
{"id":"chatcmpl-8b4773e9-a98b-41db-8163-018ed3dc65e2","object":"chat.completion","created":1720183759,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}%
```

### Deleting a Model
To delete a deployed model, use the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.

You can also remove several models at once by providing multiple model names separated by spaces:

```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b
```

### Cleanup

If you need to stop and remove the containers, you can use the following commands:

```bash
docker exec ray_head sh -c "ray stop"
docker exec ray_worker_0 sh -c "ray stop"
docker exec ray_worker_1 sh -c "ray stop"

docker stop ray_head ray_worker_0 ray_worker_1
docker rm ray_head ray_worker_0 ray_worker_1
docker network rm sllm
```