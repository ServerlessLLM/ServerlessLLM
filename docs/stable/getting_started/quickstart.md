---
sidebar_position: 1
---

# Quickstart Guide

This guide will help you get started with the basics of using ServerlessLLM. Please make sure you have installed the ServerlessLLM following the [installation guide](./installation.md).

## Run ServerlessLLM on your local machine
First, let's start a local ray cluster to run ServerlessLLM. You can start a local ray cluster by running the following command:

Start a local ray cluster with 1 head node and 1 worker node:
```bash
conda activate sllm
ray start --head --port=6379 --num-cpus=4 --num-gpus=0 \
--resources='{"control_node": 1}' --block
```

In a new terminal, start the worker node:
```bash
conda activate sllm-worker
ray start --address=0.0.0.0:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```

And start ServerlessLLM Store server. This server will use `./models` as the storage path by default.

```bash
conda activate sllm-worker
sllm-store-server
```

Expected output:
```bash
$ sllm-store-server
TODO Run server...
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20240720 08:40:59.634253 141776 server.cpp:307] Log directory already exists.
I20240720 08:40:59.671192 141776 checkpoint_store.cpp:46] Number of GPUs: 2
I20240720 08:40:59.671768 141776 checkpoint_store.cpp:48] I/O threads: 4, chunk size: 32MB
I20240720 08:40:59.797175 141776 checkpoint_store.cpp:69] GPU 0 UUID: cef23f2a-71f7-44f3-8246-5ebd870755e7
I20240720 08:40:59.951408 141776 checkpoint_store.cpp:69] GPU 1 UUID: bbd10d20-aed8-4324-8b2e-7b6e54aaca0e
I20240720 08:41:00.759124 141776 pinned_memory_pool.cpp:29] Creating PinnedMemoryPool with 1024 buffers of 33554432 bytes
I20240720 08:41:24.315564 141776 checkpoint_store.cpp:80] Memory pool created with 32GB
I20240720 08:41:24.318261 141776 server.cpp:279] Server listening on 0.0.0.0:8073
```

Now, let’s start ServerlessLLM.

First, in another new terminal, start ServerlessLLM Serve (i.e., `sllm-serve`)

```bash
conda activate sllm
sllm-serve start
```

Everything is set!

Now you have opened 4 terminals: started a local ray cluster(head node and worker node), started the ServerlessLLM Serve, and started the ServerlessLLM Store server.

Next, open another new terminal, let's deploy a model to the ServerlessLLM server. You can deploy a model by running the following command:

```bash
conda activate sllm
sllm-cli deploy --model facebook/opt-1.3b
```

Now, you can query the model by any OpenAI API client. For example, you can use the following Python code to query the model:
```bash
curl http://0.0.0.0:8343/v1/chat/completions \
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
{"id":"chatcmpl-9f812a40-6b96-4ef9-8584-0b8149892cb9","object":"chat.completion","created":1720021153,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

To delete a deployed model, use the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.

You can also remove several models at once by providing multiple model names separated by spaces:

```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b
```
