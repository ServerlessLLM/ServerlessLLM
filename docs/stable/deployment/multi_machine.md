---
sidebar_position: 2
---

# Multi-machine

This guide will help you get started with running ServerlessLLM on multiple machines using Docker containers. You'll learn how to set up a head node on one machine and connect worker nodes from different machines using Docker, ensuring proper network communication between the containers. You can extend this setup to use as many nodes as you need.

## Prerequisites

This guide requires **at least three machines**:
- One machine for the head node (no GPU required)
- At least two machines with NVIDIA GPUs to serve as worker nodes

You can add more worker machines as needed to scale out your deployment.

### For All Machines

Ensure you have the following installed and configured on all machines (both head node and worker machines):

1. **Docker**: Installed on your system. You can download it from [here](https://docs.docker.com/get-docker/).
2. **Network connectivity**: Ensure all machines can communicate with each other on the required ports (6379 for Ray, 8343 for ServerlessLLM API, and 8073 for storage service).

### For Client Machine

You'll need the following on at least one machine (which can be your local computer or the head node) to manage model deployments:

1. **ServerlessLLM CLI**: Install it using `pip install serverless-llm`.

### For Worker Machines Only

These requirements are only necessary for the worker machines that will run the models:

1. **GPUs**: At least one NVIDIA GPU is required on each worker machine. If you have multiple GPUs, you can adjust the Docker configuration accordingly.
2. **NVIDIA Docker Toolkit**: This enables Docker to utilize NVIDIA GPUs. Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Multi-Machine Setup

We'll start a head node on one machine using Docker, then add multiple worker nodes from other machines using Docker containers.

### Step 1: Start the Head Node

1. **Start the head node using Docker:**

```bash
docker run -d \
  --name sllm_head \
  -p 6379:6379 \
  -p 8343:8343 \
  -e MODE=HEAD \
  serverlessllm/sllm:latest
```

:::tip
If you don't have the ServerlessLLM Docker image locally, Docker will automatically pull it from the registry. You can also adjust the CPU and resource allocations by setting additional environment variables like `RAY_NUM_CPUS` and `RAY_RESOURCES`.
:::

2. **Verify the head node is running:**

```bash
docker logs sllm_head
```

Expected output should include:

```bash
Local node IP: <HEAD_NODE_IP>
--------------------
Ray runtime started.
--------------------
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8343 (Press CTRL+C to quit)
```

Note the `<HEAD_NODE_IP>` from the logs, which you'll use to connect worker nodes.

### Step 2: Start Worker Nodes on Different Machines

:::tip
You can adjust the memory pool size and other parameters based on the resources available on each worker machine.
:::

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node machine.

1. **On the first worker machine, create a directory for model storage:**

```bash
mkdir -p /path/to/your/models
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

2. **Start the first worker node:**

```bash
docker run -d \
  --name sllm_worker_0 \
  -p 8073:8073 \
  --gpus '"device=0"' \
  -e WORKER_ID=0 \
  -e STORAGE_PATH=/models \
  -e MODE=WORKER \
  -e RAY_HEAD_ADDRESS=<HEAD_NODE_IP>:6379 \
  -v ${MODEL_FOLDER}:/models \
  serverlessllm/sllm:latest \
  --mem-pool-size 4GB --registration-required true
```

3. **On the second worker machine, repeat the setup:**

```bash
mkdir -p /path/to/your/models
export MODEL_FOLDER=/path/to/your/models

docker run -d \
  --name sllm_worker_1 \
  -p 8073:8073 \
  --gpus '"device=0"' \
  -e WORKER_ID=1 \
  -e STORAGE_PATH=/models \
  -e MODE=WORKER \
  -e RAY_HEAD_ADDRESS=<HEAD_NODE_IP>:6379 \
  -v ${MODEL_FOLDER}:/models \
  serverlessllm/sllm:latest \
  --mem-pool-size 4GB --registration-required true
```

You can continue adding more worker nodes by repeating the above steps on additional machines, specifying a unique `WORKER_ID` for each node (2, 3, 4, etc.).

4. **Verify worker nodes are connected:**

On each worker machine, check the container logs:

```bash
docker logs sllm_worker_0  # Use appropriate worker container name
```

Expected output should include:

```bash
INFO 12-31 17:09:35 cli.py:58] Starting gRPC server
INFO 12-31 17:09:35 server.py:34] StorageServicer: storage_path=/models, mem_pool_size=4294967296, num_thread=4, chunk_size=33554432, registration_required=true
INFO 12-31 17:09:38 server.py:243] Starting gRPC server on 0.0.0.0:8073
```

### Step 3: Configure Alternative Network Setup (if needed)

If you prefer to use host networking for simpler network configuration, you can use the following alternative approach:

**For the head node:**

```bash
docker run -d \
  --name sllm_head \
  --network host \
  -e MODE=HEAD \
  serverlessllm/sllm:latest
```

**For worker nodes:**

```bash
docker run -d \
  --name sllm_worker_0 \
  --network host \
  --gpus '"device=0"' \
  -e WORKER_ID=0 \
  -e STORAGE_PATH=/models \
  -e MODE=WORKER \
  -e RAY_HEAD_ADDRESS=<HEAD_NODE_IP>:6379 \
  -v ${MODEL_FOLDER}:/models \
  serverlessllm/sllm:latest \
  --mem-pool-size 4GB --registration-required true
```

### Step 4: Use `sllm-cli` to manage models

#### Configure the Environment

**On any machine with `sllm-cli` installed, set the `LLM_SERVER_URL` environment variable:**

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

```bash
export LLM_SERVER_URL=http://<HEAD_NODE_IP>:8343
```

#### Deploy a Model Using `sllm-cli`

```bash
sllm-cli deploy --model facebook/opt-1.3b
```

> Note: This command will spend some time downloading the model from the Hugging Face Model Hub. You can use any model from the [Hugging Face Model Hub](https://huggingface.co/models) by specifying the model name in the `--model` argument.

Expected output:

```bash
INFO 07-24 06:51:32 deploy.py:83] Model registered successfully.
```

#### Delete a Deployed Model Using `sllm-cli`

- To delete a deployed model, use the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.

- You can also remove several models at once by providing multiple model names separated by spaces:

```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b
```

### Step 5: Query the Model Using OpenAI API Client

**You can query the model using any OpenAI API client. For example, use the following command:**

**Make sure the model is successfully deployed before querying.**

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

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

```json
{"id":"chatcmpl-23d3c0e5-70a0-4771-acaf-bcb2851c6ea6","object":"chat.completion","created":1721706121,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

## Troubleshooting

### Network Issues

1. **Connection refused errors**: Ensure that firewalls on all machines allow traffic on ports 6379, 8343, and 8073.

2. **Ray cluster connection issues**: Verify that the head node IP address is correct and that the Ray port (6379) is accessible from worker machines.

3. **GPU access issues**: Make sure the NVIDIA Docker toolkit is properly installed and that the `--runtime=nvidia` flag is used for worker containers (if it is not set as default).

### Container Management

- **View running containers**: `docker ps`

To stop and remove all ServerlessLLM containers:

1. **Stop all containers:**

```bash
# On head node machine
docker stop sllm_head
docker rm sllm_head

# On each worker machine
docker stop sllm_worker_0  # Use appropriate container name (sllm_worker_1, sllm_worker_2, etc.)
docker rm sllm_worker_0
```

2. **Optional: Remove the Docker image:**

```bash
docker rmi serverlessllm/sllm:latest
```
