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
2. **Network connectivity**: Ensure all machines can communicate with each other on the required ports (6379 for Ray, 8343 for ServerlessLLM API, and 8073 for storage service).

:::tip
The **ServerlessLLM CLI** (`pip install serverless-llm`) can be installed on any machine that needs to manage model deployments. This could be your local computer or any machine within the cluster that can connect to the head node.
:::

### For Worker Machines Only

These requirements are only necessary for the worker machines that will run the models:

1. **GPUs**: At least one NVIDIA GPU is required on each worker machine. If you have multiple GPUs, you can adjust the Docker configuration accordingly.
2. **NVIDIA Docker Toolkit**: This enables Docker to utilize NVIDIA GPUs. Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Multi-Machine Setup

We'll start a head node on one machine using Docker, then add a worker node from another machine using Docker containers with host networking.

### Step 1: Start the Head Node

1. **Start the head node using Docker:**

```bash
# Get the machine's IP address that will be accessible to other machines
export HEAD_IP=$(hostname -I | awk '{print $1}')
echo "Head node IP address: $HEAD_IP"

docker run -d \
  --name sllm_head \
  --network host \
  -e MODE=HEAD \
  -e RAY_NODE_IP=$HEAD_IP \
  serverlessllm/sllm:latest
```

:::important
For multi-machine setups, setting the `RAY_NODE_IP` is critical. It should be set to an IP address that is accessible from all worker machines. The command above attempts to automatically determine your machine's primary IP, but in complex network environments, you may need to specify it manually.

If your machine has multiple network interfaces, ensure you use the IP that other machines in your network can access.
:::

:::tip
If you don't have the ServerlessLLM Docker image locally, Docker will automatically pull it from the registry. You can also adjust the CPU and resource allocations by setting additional environment variables like `RAY_NUM_CPUS` and `RAY_RESOURCES`.
:::

2. **Verify the head node is running and note the external IP:**

```bash
docker logs sllm_head
```

Expected output should include:

```bash
> docker logs sllm_head
...
2025-05-29 14:29:46,211	INFO scripts.py:744 -- Local node IP: 129.215.164.107
...
(SllmController pid=380) INFO 05-29 14:29:53 controller.py:59] Starting store manager
(SllmController pid=380) INFO 05-29 14:29:56 controller.py:68] Starting scheduler
(StoreManager pid=417) INFO 05-29 14:29:56 store_manager.py:226] Initializing store manager
(StoreManager pid=417) INFO 05-29 14:29:56 store_manager.py:237] Initializing cluster and collecting hardware info
(StoreManager pid=417) ERROR 05-29 14:29:56 store_manager.py:242] No worker nodes found
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8343 (Press CTRL+C to quit)
(FcfsScheduler pid=456) INFO 05-29 14:29:56 fcfs_scheduler.py:54] Starting FCFS scheduler
(FcfsScheduler pid=456) INFO 05-29 14:29:56 fcfs_scheduler.py:111] Starting control loop
```

Make note of the IP address shown in the logs. This is the address that worker nodes will use to connect to the head node.

### Step 2: Start Worker Node on a Different Machine

:::tip
You can adjust the memory pool size and other parameters based on the resources available on your worker machine.
:::

1. **On the worker machine, create a directory for model storage:**

```bash
mkdir -p /path/to/your/models
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

2. **Start the worker node:**

```bash
# Replace with the actual IP address of the head node from the previous step
# DO NOT copy-paste this line directly - update with your actual head node IP
export HEAD_IP=<HEAD_NODE_IP>
```

```bash
# Get the worker machine's IP address that will be accessible to the head node
export WORKER_IP=$(hostname -I | awk '{print $1}')
echo "Worker node IP address: $WORKER_IP"

docker run -d \
  --name sllm_worker_0 \
  --network host \
  --gpus '"device=0"' \
  -e WORKER_ID=0 \
  -e STORAGE_PATH=/models \
  -e MODE=WORKER \
  -e RAY_HEAD_ADDRESS=${HEAD_IP}:6379 \
  -e RAY_NODE_IP=$WORKER_IP \
  -v ${MODEL_FOLDER}:/models \
  serverlessllm/sllm:latest \
  --mem-pool-size 4GB --registration-required true
```

:::important
For multi-machine setups, setting the `RAY_NODE_IP` on worker nodes is just as critical as on the head node. It should be set to an IP address that is accessible from the head node. Without this, workers might report internal Docker IPs that aren't accessible across machines.

Make sure to replace `192.168.1.100` with the actual IP address of your head node that you noted earlier.
:::

3. **Verify worker node is connected:**

On the worker machine, check if the worker has properly connected to the Ray cluster:

```bash
docker exec -it sllm_worker_0 bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate worker && ray status"
```

Expected output should include both the head node and worker node resources:

```bash
> docker exec -it sllm_worker_0 bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate worker && ray status"
======== Autoscaler status: 2025-05-29 14:42:30.434645 ========
Node status
---------------------------------------------------------------
Active:
 1 node_f0a8e97ca64c64cebd551f469a38d0d66ce304f7cc1cc9696fe33cf3
 1 node_3b7db178afb8bdb16460d0cb6463dc7b9b3afbcc00753c3be110c9b3
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 3.0/52.0 CPU
 0.0/1.0 GPU
 0.30000000000000004/1.0 control_node
 0B/526.36GiB memory
 0B/18.63GiB object_store_memory
 0.0/1.0 worker_id_0
 0.0/1.0 worker_node

Demands:
 (no resource demands)
```

This output confirms that both the head node and worker node are properly connected and their resources are recognized by the Ray cluster.

:::tip
**Adding more worker nodes:** You can add more worker nodes by repeating Step 2 on additional machines with GPUs. Just make sure to:
1. Use a unique `WORKER_ID` for each worker (1, 2, 3, etc.)
2. Point each worker to the same head node IP address
3. Ensure each worker has its own `RAY_NODE_IP` set correctly
:::

### Step 3: Use `sllm-cli` to manage models

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

### Step 4: Query the Model Using OpenAI API Client

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

#### Delete a Deployed Model Using `sllm-cli`

When you're done using a model, you can delete it:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.

## Clean Up

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

:::tip
If you don't have the ServerlessLLM Docker image locally, Docker will automatically pull it from the registry. You can also adjust the CPU and resource allocations by setting additional environment variables like `RAY_NUM_CPUS` and `RAY_RESOURCES`.
:::

## Troubleshooting

### Network Issues

1. **Connection refused errors**: Ensure that firewalls on all machines allow traffic on ports 6379, 8343, and 8073.

2. **Ray cluster connection issues**:
   - Verify that the head node IP address is correct and that the Ray port (6379) is accessible from worker machines
   - Ensure both head and worker nodes have their `RAY_NODE_IP` set to an IP address that is accessible from other machines
   - Check that you're not using private Docker network IPs (typically 172.x.x.x) which aren't accessible across machines

3. **Workers can't connect to head node**:
   - Make sure the `RAY_HEAD_ADDRESS` points to the external IP of the head node, not localhost or an internal Docker IP
   - Verify network connectivity with `ping` or `telnet` from worker machines to the head node IP on port 6379

4. **GPU access issues**: Make sure the NVIDIA Docker toolkit is properly installed and that the `--gpus` flag is used for worker containers.

### Container Management

- **View running containers**: `docker ps`