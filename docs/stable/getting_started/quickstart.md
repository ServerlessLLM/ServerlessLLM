---
sidebar_position: 1
---

# Quickstart Guide

This guide will help you get started with the basics of using ServerlessLLM. Please make sure you have installed the ServerlessLLM following the [installation guide](./installation.md).

## Local test
First, let's start a local ray cluster to test the ServerlessLLM. You can start a local ray cluster by running the following command:

Start a local ray cluster with 1 head node and 1 worker node:
```bash
conda activate sllm
ray start --head --port=6379 --num-cpus=4 --num-gpus=0 \
--resources='{"control_node": 1}' --block
```

In a new terminal, start the worker node:
```bash
conda activate sllm
ray start --address=localhost:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```

Now, let’s start ServerlessLLM.

First, start ServerlessLLM Serve (i.e., `sllm-serve`)

```bash
conda activate sllm
sllm-serve start
```

Next start ServerlessLLM Store server. This server will use `./models` as the storage path by default.

```bash
conda activate sllm
sllm-store-server
```

Everything is set!

Next, let's deploy a model to the ServerlessLLM server. You can deploy a model by running the following command:

```bash
conda activate sllm
sllm-cli deploy --model facebook/opt-1.3b
```

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
```json
{"id":"chatcmpl-9f812a40-6b96-4ef9-8584-0b8149892cb9","object":"chat.completion","created":1720021153,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

To delete a deployed model, use the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

This will remove the specified model from the ServerlessLLM server.