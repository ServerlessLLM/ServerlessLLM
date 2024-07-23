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

### Step 2: Start Worker Nodes on Different Machines

:::tip
You can adjust the number of CPUs and GPUs based on the resources available on each machine.
:::

1. **On the first worker machine, activate the `sllm` environment and connect to the head node:**

```bash
conda activate sllm
ray start --address=<HEAD_NODE_IP>:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```

2. **On the second worker machine, activate the `sllm` environment and connect to the head node:**

```bash
conda activate sllm
ray start --address=<HEAD_NODE_IP>:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_1": 1}' --block
```


> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

You can continue adding more worker nodes by repeating the above steps on additional machines, specifying a unique `worker_id` for each node.

### Step 3: Start ServerlessLLM Serve on the Head Node

1. **On the head node machine, start ServerlessLLM Serve:**

```bash
conda activate sllm
sllm-serve start
```

### Step 4: Start ServerlessLLM Store Server on the Worker Nodes

1. **On each worker node machine, start the ServerlessLLM Store server:**

```bash
conda activate sllm
sllm-store-server
```

### Step 5: Deploy a Model Using `sllm-cli`

1. **On any machine, open a new terminal, activate the `sllm` environment, and set the `LLM_SERVER_URL` environment variable:**

```bash
conda activate sllm
export LLM_SERVER_URL=http://<HEAD_NODE_IP>:8343/
```

2. **Deploy a model to the ServerlessLLM server using the `sllm-cli`:**

```bash
sllm-cli deploy --model facebook/opt-1.3b
```

> Note: This command will spend some time downloading the model from the Hugging Face Model Hub. You can use any model from the [Hugging Face Model Hub](https://huggingface.co/models) by specifying the model name in the `--model` argument.

### Step 6: Query the Model Using OpenAI API Client

1. **You can query the model by any OpenAI API client. For example, you can use the following command to query the model:**

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

> Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.

Expected output:

```json
{"id":"chatcmpl-23d3c0e5-70a0-4771-acaf-bcb2851c6ea6","object":"chat.completion","created":1721706121,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

### Step 7: Delete a Deployed Model

1. **To delete a deployed model, use the following command:**

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.

2. **You can also remove several models at once by providing multiple model names separated by spaces:**

```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b
```

