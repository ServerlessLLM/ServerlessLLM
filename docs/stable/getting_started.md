---
sidebar_position: 1
---

# Getting Started

This guide demonstrates how to quickly set up a local ServerlessLLM cluster using Docker Compose on a single machine. We will initialize a minimal cluster, consisting of a head node and a single worker node. Then, we'll deploy a model using the `sllm` and query the deployment through an OpenAI-compatible API.

:::note
We strongly recommend using Docker (Compose) to manage your ServerlessLLM cluster, whether you are using ServerlessLLM for testing or development. However, if Docker is not a viable option for you, please refer to the [deploy from scratch guide](./deployment/single_machine.md).
:::

## Prerequisites

Before you begin, ensure you have the following installed and configured:

1.  **Docker**: Installed on your system. You can download it from [here](https://docs.docker.com/get-docker/).
2.  **ServerlessLLM CLI**: Installed on your system. Install it using `pip install serverless-llm`.
3.  **GPUs**: At least one NVIDIA GPU is required. If you have multiple GPUs, you can adjust the `docker-compose.yml` file accordingly.
4.  **NVIDIA Docker Toolkit**: This enables Docker to utilize NVIDIA GPUs. Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Start the ServerlessLLM Cluster

We will use Docker Compose to simplify the ServerlessLLM setup process.

### Step 1: Download the Docker Compose File

Download the `docker-compose.yml` file from the ServerlessLLM repository:

```bash
# Create a directory for the ServerlessLLM Docker setup
mkdir serverless-llm-docker && cd serverless-llm-docker

# Download the docker-compose.yml file
curl -O https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/examples/docker/docker-compose.yml

# Alternatively, you can use wget:
# wget https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/examples/docker/docker-compose.yml
```

### Step 2: Configuration

Create a directory on your host machine to store models. Then, set the `MODEL_FOLDER` environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you intend to store the models. This directory will be mounted into the Docker containers.

### Step 3: Start the Services

Start the ServerlessLLM services using Docker Compose:

```bash
docker compose up -d
```

This command will start the Ray head node and a worker node as defined in the `docker-compose.yml` file.

Verify that the services are ready:

```bash
docker logs sllm_head
```

Ensure the services are ready before proceeding. You should see output similar to the following:

```bash
...
(SllmController pid=1435) INFO 05-26 15:40:49 controller.py:68] Starting scheduler
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8343 (Press CTRL+C to quit)
(FcfsScheduler pid=1604) INFO 05-26 15:40:49 fcfs_scheduler.py:54] Starting FCFS scheduler
(FcfsScheduler pid=1604) INFO 05-26 15:40:49 fcfs_scheduler.py:111] Starting control loop
```

## Deploy a Model Using sllm click

Set the `LLM_SERVER_URL` environment variable:

```bash
export LLM_SERVER_URL=http://127.0.0.1:8343
```

Deploy a model to the ServerlessLLM cluster using the `sllm`:

```bash
sllm deploy --model facebook/opt-1.3b
```
> Note: This command will take some time to download the model from the Hugging Face Model Hub.
> You can use any model from the [Hugging Face Model Hub](https://huggingface.co/models) by specifying its name in the `--model` argument.

Expected output:

```plaintext
INFO 08-01 07:38:12 deploy.py:36] Deploying model facebook/opt-1.3b with default configuration.
INFO 08-01 07:39:00 deploy.py:49] Model registered successfully.
```

## Query the Model

You can now query the model using any OpenAI API client. For example, use the following `curl` command:
```bash
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

Expected output:

```plaintext
{"id":"chatcmpl-8b4773e9-a98b-41db-8163-018ed3dc65e2","object":"chat.completion","created":1720183759,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}%
```

## Clean Up
To delete a deployed model, execute the following command:

```bash
sllm delete facebook/opt-1.3b
```

This command removes the specified model from the ServerlessLLM server.

To stop the ServerlessLLM services, use the following command:
```bash
docker compose down
```