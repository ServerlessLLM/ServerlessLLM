---
sidebar_position: 3
---

# Multi-Machine Setup Guide

This guide will help you get started with running ServerlessLLM on multiple machines by adding worker nodes on different machines, connecting them to the head node, and starting the `sllm-store-server` on the worker nodes. You can extend this setup to use as many nodes as you need. Please make sure you have installed the ServerlessLLM following the [installation guide](./installation.md) on all machines.

## Multi-Machine Setup

First, let's start a head node on the main machine and then add multiple worker nodes from other machines.

### Step 1: Start the Head Node on the Main Machine

1. **Activate the `sllm` environment and start the head node:**

```bash
conda activate sllm
ray start --head --port=6379 --num-cpus=12 --num-gpus=0 \
--resources='{"control_node": 1}' --block
```

Expected output:

```bash
Local node IP: 129.215.164.142

--------------------
Ray runtime started.
--------------------
```

Here `Local node IP` is the IP address of the head node, which we would use as `<HEAD_NODE_IP>` in the following steps.

### Step 2: Start Worker Nodes on Different Machines

:::tip
You can adjust the number of CPUs and GPUs based on the resources available on each machine.
:::

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

1. **On the first worker machine, activate the `sllm-worker` environment and connect to the head node:**

```bash
conda activate sllm-worker
ray start --address=<HEAD_NODE_IP>:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```

2. **On the second worker machine, activate the `sllm-worker` environment and connect to the head node:**

```bash
conda activate sllm-worker
ray start --address=<HEAD_NODE_IP>:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_1": 1}' --block
```

You can continue adding more worker nodes by repeating the above steps on additional machines, specifying a unique `worker_id` for each node.

### Step 3: Start ServerlessLLM Store Server on the Worker Nodes

1. **On EACH worker node machine, start the ServerlessLLM Store server:**

```bash
conda activate sllm-worker
sllm-store-server
```

Expected output:

```bash
TODO Run server...
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20240724 06:46:25.054241 1337444 server.cpp:290] Log directory already exists.
I20240724 06:46:25.199916 1337444 checkpoint_store.cpp:29] Number of GPUs: 4
I20240724 06:46:25.200362 1337444 checkpoint_store.cpp:31] I/O threads: 4, chunk size: 32MB
I20240724 06:46:25.326860 1337444 checkpoint_store.cpp:52] GPU 0 UUID: c9938b31-33b0-e02f-24c5-88bd6fbe19ad
I20240724 06:46:25.472143 1337444 checkpoint_store.cpp:52] GPU 1 UUID: 3f4f72ef-ed7f-2ddb-e454-abcc6c0330b0
I20240724 06:46:25.637110 1337444 checkpoint_store.cpp:52] GPU 2 UUID: 99b39a1b-5fdd-1acb-398a-426672ebc1a8
I20240724 06:46:25.795079 1337444 checkpoint_store.cpp:52] GPU 3 UUID: c164f9d9-f157-daeb-d7be-5c98029c2a2b
I20240724 06:46:25.795164 1337444 pinned_memory_pool.cpp:12] Creating PinnedMemoryPool with 1024 buffers of 33554432 bytes
I20240724 06:46:40.843920 1337444 checkpoint_store.cpp:63] Memory pool created with 32GB
I20240724 06:46:40.845937 1337444 server.cpp:262] Server listening on 0.0.0.0:8073
```

### Step 4: Start ServerlessLLM Serve on the Head Node

1. **On the head node machine, start ServerlessLLM Serve:**

```bash
conda activate sllm
sllm-serve start
```

Expected output:

```bash
2024-07-24 06:49:03,513 INFO worker.py:1540 -- Connecting to existing Ray cluster at address: 129.215.164.142:6379...
2024-07-24 06:49:03,522 INFO worker.py:1724 -- Connected to Ray cluster.
INFO:     Started server process [1339357]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8343 (Press CTRL+C to quit)
(FcfsScheduler pid=1339527) INFO 07-24 06:49:06 fcfs_scheduler.py:55] Starting FCFS scheduler
(FcfsScheduler pid=1339527) INFO 07-24 06:49:06 fcfs_scheduler.py:95] Starting control loop
(FcfsScheduler pid=1339527) INFO 07-24 06:49:06 fcfs_scheduler.py:55] Starting FCFS scheduler
(FcfsScheduler pid=1339527) INFO 07-24 06:49:06 fcfs_scheduler.py:95] Starting control loop
```

### Step 5: Use `sllm-cli` to manage models

#### Configure the Environment
**On any machine, open a new terminal, activate the `sllm` environment, and set the `LLM_SERVER_URL` environment variable:**

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

```bash
conda activate sllm
export LLM_SERVER_URL=http://<HEAD_NODE_IP>:8343/
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


### Step 6: Query the Model Using OpenAI API Client

1. **You can query the model by any OpenAI API client. For example, you can use the following command to query the model:**

**Make sure the model is successfully deployed before querying.**

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

```bash
curl http://<HEAD_NODE_IP>:8343/v1/chat/completions \
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
