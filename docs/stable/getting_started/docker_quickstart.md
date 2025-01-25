---
sidebar_position: 2
---

# Docker Quickstart Guide

This guide shows how to quickly set up a local ServerlessLLM cluster using Docker Compose. We will start a minimal cluster with a head node and one worker node, deploy and query a model using the `sllm-cli`.

## Pre-requisites

Before you begin, make sure you have the following:

1. **Docker**: Installed on your system. You can download it from [here](https://docs.docker.com/get-docker/).
2. **ServerlessLLM CLI**: Installed on your system. You can install it using `pip install serverless-llm`.
1. **GPUs**: At least one NVIDIA GPU is necessary. If you have more GPUs, you can adjust the `docker-compose.yml` file accordingly.
2. **NVIDIA Docker Toolkit**: This allows Docker to use NVIDIA GPUs. Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Run ServerlessLLM using Docker Compose

We will use docker compose to simplify the setup of ServerlessLLM. The `docker-compose.yml` file is located in the `examples/docker/` directory of the ServerlessLLM repository.

### Step 1: Clone the ServerlessLLM Repository

If you haven't already, clone the ServerlessLLM repository:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/examples/docker/
```

### Step 2:  Configuration

Create a directory on your host machine where models will be stored and set the MODEL_FOLDER environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace /path/to/your/models with the actual path where you want to store the models.

### Step 3: Start the Services

Start the ServerlessLLM services using docker compose:

```bash
docker compose up -d
```

This command will start the Ray head node and two worker nodes defined in the `docker-compose.yml` file.

### Step 4: Deploy a Model Using sllm-cli

Open a new terminal, activate the `sllm` environment, and set the `LLM_SERVER_URL` environment variable:

```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343/
```

Deploy a model to the ServerlessLLM server using the `sllm-cli`:

```bash
sllm-cli deploy --model facebook/opt-1.3b
```
> Note: This command will spend some time downloading the model from the Hugging Face Model Hub.
> You can use any model from the [Hugging Face Model Hub](https://huggingface.co/models) by specifying the model name in the `--model` argument.

Expected output:

```plaintext
INFO 08-01 07:38:12 deploy.py:36] Deploying model facebook/opt-1.3b with default configuration.
INFO 08-01 07:39:00 deploy.py:49] Model registered successfully.
```

### Step 5: Query the Model

Now, you can query the model by any OpenAI API client. For example, you can use the following Python code to query the model:
```bash
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

Expected output:

```plaintext
{"id":"chatcmpl-8b4773e9-a98b-41db-8163-018ed3dc65e2","object":"chat.completion","created":1720183759,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}%
```

### Step 6: Clean Up
To delete a deployed model, use the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.

To stop the ServerlessLLM services, use the following command:
```bash
docker compose down
```