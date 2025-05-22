---
sidebar_position: 2
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
export CUDA_VISIBLE_DEVICES=0
ray start --address=0.0.0.0:6379 --num-cpus=4 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```

And start ServerlessLLM Store server. This server will use `./models` as the storage path by default.

```bash
conda activate sllm-worker
export CUDA_VISIBLE_DEVICES=0
sllm-store start
```

Expected output:
```bash
$ sllm-store start
INFO 12-31 17:13:23 cli.py:58] Starting gRPC server
INFO 12-31 17:13:23 server.py:34] StorageServicer: storage_path=./models, mem_pool_size=4294967296, num_thread=4, chunk_size=33554432, registration_required=False
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20241231 17:13:23.947276 2165054 checkpoint_store.cpp:41] Number of GPUs: 1
I20241231 17:13:23.947299 2165054 checkpoint_store.cpp:43] I/O threads: 4, chunk size: 32MB
I20241231 17:13:23.947309 2165054 checkpoint_store.cpp:45] Storage path: "./models"
I20241231 17:13:24.038651 2165054 checkpoint_store.cpp:71] GPU 0 UUID: c9938b31-33b0-e02f-24c5-88bd6fbe19ad
I20241231 17:13:24.038700 2165054 pinned_memory_pool.cpp:29] Creating PinnedMemoryPool with 128 buffers of 33554432 bytes
I20241231 17:13:25.557906 2165054 checkpoint_store.cpp:83] Memory pool created with 4GB
INFO 12-31 17:13:25 server.py:243] Starting gRPC server on 0.0.0.0:8073
```

Now, let's start ServerlessLLM.

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

This will download the model from HuggingFace, if you want load the model from local path, you can use `config.json`, see [here](../cli/cli_api.md#example-configuration-file-configjson) for details.

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
```json
{"id":"chatcmpl-9f812a40-6b96-4ef9-8584-0b8149892cb9","object":"chat.completion","created":1720021153,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

To delete a deployed model, use the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.