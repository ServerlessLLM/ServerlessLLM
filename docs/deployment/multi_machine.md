---
sidebar_position: 2
---

# Multi-machine

This guide will help you get started with running ServerlessLLM on multiple machines using Docker containers. You'll learn how to set up a head node on one machine and connect worker nodes from different machines using Docker, ensuring proper network communication between the containers. You can extend this setup to use as many nodes as you need.

## Prerequisites

This guide requires **two machines**:
- One machine for the head node (no GPU required)
- One machine with an NVIDIA GPU to serve as the worker node

You can add more worker machines with GPUs as needed to scale out your deployment.

### For All Machines

Ensure you have the following installed and configured on all machines (both head node and worker machines):

1. **Docker**: Installed on your system. You can download it from [here](https://docs.docker.com/get-docker/).
2. **Network connectivity**: Ensure all machines can communicate with each other on the required ports (8000 for Pylet, 8343 for ServerlessLLM API, and 8080-8179 for vLLM instance ports).

:::tip
The **ServerlessLLM CLI** (`pip install serverless-llm`) can be installed on any machine that needs to manage model deployments. This could be your local computer or any machine within the cluster that can connect to the head node.
:::

### For Worker Machines Only

These requirements are only necessary for the worker machines that will run the models:

1. **GPUs**: At least one NVIDIA GPU is required on each worker machine. If you have multiple GPUs, you can adjust the Docker configuration accordingly.
2. **NVIDIA Docker Toolkit**: This enables Docker to utilize NVIDIA GPUs. Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Multi-Machine Setup

We'll start a Pylet head (cluster manager) on one machine, then add a Pylet worker node from another machine, and finally start the SLLM head (control plane) using Docker containers with host networking.

### Step 1: Start the Pylet Head Node

The Pylet head is the cluster manager that coordinates workers and schedules GPU instances.

1. **Start the Pylet head node using Docker:**

```bash
# Get the machine's IP address that will be accessible to other machines
export HEAD_IP=$(hostname -I | awk '{print $1}')
echo "Head node IP address: $HEAD_IP"

docker run -d \
  --name pylet_head \
  --network host \
  python:3.10-slim \
  sh -c "pip install pylet httpx && pylet start"
```

:::important
For multi-machine setups, make note of the `HEAD_IP`. This IP address must be accessible from all worker machines. The command above attempts to automatically determine your machine's primary IP, but in complex network environments, you may need to specify it manually.

If your machine has multiple network interfaces, ensure you use the IP that other machines in your network can access.
:::

:::tip
If you don't have the Docker image locally, Docker will automatically pull it from the registry.
:::

2. **Verify the Pylet head node is running:**

```bash
curl http://localhost:8000/workers
```

Expected output should show an empty worker list initially:

```json
[]
```

Make note of the IP address of this machine. This is the address that worker nodes will use to connect to the Pylet head.

### Step 2: Start Pylet Worker Node on a Different Machine

:::tip
You can adjust the GPU units and other parameters based on the resources available on your worker machine.
:::

1. **On the worker machine, create a directory for model storage:**

```bash
mkdir -p /path/to/your/models
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

2. **Start the Pylet worker node:**

```bash
# Replace with the actual IP address of the Pylet head node from the previous step
# DO NOT copy-paste this line directly - update with your actual head node IP
export HEAD_IP=<HEAD_NODE_IP>
```

```bash
docker run -d \
  --name pylet_worker_0 \
  --network host \
  --gpus '"device=0"' \
  -e MODE=PYLET_WORKER \
  -e PYLET_HEAD=${HEAD_IP}:8000 \
  -e GPU_UNITS=1 \
  -e STORAGE_PATH=/models \
  -v ${MODEL_FOLDER}:/models \
  serverlessllm/sllm:latest
```

:::important
For multi-machine setups, ensure the `PYLET_HEAD` environment variable points to the correct IP address and port (8000) of the Pylet head node.

Make sure to replace `<HEAD_NODE_IP>` with the actual IP address of your Pylet head node that you noted earlier.
:::

3. **Verify worker node is connected:**

On the head node machine, check if the worker has properly connected to the Pylet cluster:

```bash
curl http://localhost:8000/workers
```

Expected output should include the worker node:

```json
[
  {
    "id": "worker-0",
    "gpu_units": 1,
    "status": "ready"
  }
]
```

This output confirms that the worker node is properly connected and its resources are recognized by the Pylet cluster.

:::tip
**Adding more worker nodes:** You can add more worker nodes by repeating Step 2 on additional machines with GPUs. Just make sure to:
1. Use a unique container name for each worker (e.g., `pylet_worker_1`, `pylet_worker_2`, etc.)
2. Point each worker to the same Pylet head IP address
3. Adjust `GPU_UNITS` based on the number of GPUs available on each worker
:::

### Step 3: Start the SLLM Head Node

The SLLM head is the control plane that provides the API gateway, load balancing, and autoscaling.

1. **On the same machine as the Pylet head (or another machine), start the SLLM head:**

```bash
# Replace with the actual IP address of the Pylet head node
export HEAD_IP=<HEAD_NODE_IP>

docker run -d \
  --name sllm_head \
  --network host \
  -e MODE=HEAD \
  -e PYLET_ENDPOINT=http://${HEAD_IP}:8000 \
  -e STORAGE_PATH=/models \
  -v ${MODEL_FOLDER}:/models \
  serverlessllm/sllm:latest
```

2. **Verify the SLLM head is running:**

```bash
docker logs sllm_head
```

Expected output should include:

```bash
Starting SLLM head (v1-beta)...
  Host: 0.0.0.0
  Port: 8343
  Pylet: http://<HEAD_NODE_IP>:8000
  Database: /var/lib/sllm/state.db
  Storage: /models
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8343 (Press CTRL+C to quit)
```

### Step 4: Use `sllm` to manage models

#### Configure the Environment

**On any machine with `sllm` installed, set the `LLM_SERVER_URL` environment variable:**

> Replace `<HEAD_NODE_IP>` with the actual IP address of the SLLM head node.

```bash
export LLM_SERVER_URL=http://<HEAD_NODE_IP>:8343
```

#### Deploy a Model Using `sllm`

```bash
sllm deploy --model facebook/opt-1.3b --backend vllm
```

> Note: This command will spend some time downloading the model from the Hugging Face Model Hub. You can use any model from the [Hugging Face Model Hub](https://huggingface.co/models) by specifying the model name in the `--model` argument.

Expected output:

```bash
INFO 07-24 06:51:32 deploy.py:83] Model registered successfully.
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
        "backend": "vllm",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```

Expected output:

```json
{"id":"chatcmpl-23d3c0e5-70a0-4771-acaf-bcb2851c6ea6","object":"chat.completion","created":1721706121,"model":"facebook/opt-1.3b:vllm","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

#### Delete a Deployed Model Using `sllm`

When you're done using a model, you can delete it:

```bash
sllm delete facebook/opt-1.3b --backend vllm
```

This will remove the specified model from the ServerlessLLM server.

## Clean Up

To stop and remove all ServerlessLLM containers:

1. **Stop all containers:**

```bash
# On head node machine
docker stop sllm_head pylet_head
docker rm sllm_head pylet_head

# On each worker machine
docker stop pylet_worker_0  # Use appropriate container name (pylet_worker_1, pylet_worker_2, etc.)
docker rm pylet_worker_0
```

2. **Optional: Remove the Docker image:**

```bash
docker rmi serverlessllm/sllm:latest
```

## Troubleshooting

### Network Issues

1. **Connection refused errors**: Ensure that firewalls on all machines allow traffic on ports 8000 (Pylet), 8343 (SLLM API), and 8080-8179 (vLLM instance ports).

2. **Pylet cluster connection issues**:
   - Verify that the Pylet head IP address is correct and that port 8000 is accessible from worker machines
   - Check that you're not using private Docker network IPs (typically 172.x.x.x) which aren't accessible across machines

3. **Workers can't connect to Pylet head**:
   - Make sure the `PYLET_HEAD` environment variable points to the external IP of the Pylet head node, not localhost or an internal Docker IP
   - Verify network connectivity with `ping` or `curl http://<HEAD_IP>:8000/workers` from worker machines

4. **GPU access issues**: Make sure the NVIDIA Docker toolkit is properly installed and that the `--gpus` flag is used for worker containers.

### Container Management

- **View running containers**: `docker ps`
- **Check Pylet cluster status**: `curl http://localhost:8000/workers`
- **Check Pylet instances**: `curl http://localhost:8000/instances`