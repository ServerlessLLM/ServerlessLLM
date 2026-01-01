---
sidebar_position: 1
---

# Single machine (from scratch)

This guide provides instructions for setting up ServerlessLLM from scratch on a single machine. This 'from scratch' approach means you will manually initialize and manage the Pylet cluster components. It involves using multiple terminal sessions to run the head and worker processes on the same physical machine, effectively simulating a multi-node deployment locally.

:::note
We strongly recommend using Docker (Compose) as detailed in the [Docker Compose guide](../getting_started.md). Docker provides a smoother and generally easier setup process. Follow this guide only if Docker is not a suitable option for your environment.
:::

## Installation

### Requirements

Ensure your system meets the following prerequisites:

-   **OS**: Ubuntu 20.04
-   **Python**: 3.10
-   **GPU**: NVIDIA GPU with compute capability 7.0 or higher

### Installing with pip

Follow these steps to install ServerlessLLM using pip:

**Create the head environment:**

```bash
# Create and activate a virtual environment (uv recommended)
uv venv sllm-head && source sllm-head/bin/activate

# Install ServerlessLLM and Pylet
uv pip install serverless-llm pylet
```

**Create the worker environment:**

```bash
# Create and activate a virtual environment (uv recommended)
uv venv sllm-worker && source sllm-worker/bin/activate

# Install ServerlessLLM (worker version) and its store component
uv pip install "serverless-llm[worker]" serverless-llm-store
```

:::note
If you plan to integrate vLLM with ServerlessLLM, a patch needs to be applied to the vLLM repository. For detailed instructions, please refer to the [vLLM Patch](#vllm-patch) section.
:::

### Installing from Source

To install ServerlessLLM from source, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/ServerlessLLM/ServerlessLLM.git
    cd ServerlessLLM
    ```

2.  Create the head environment:
    ```bash
    # Create and activate a virtual environment (uv recommended)
    uv venv sllm-head && source sllm-head/bin/activate

    # Install ServerlessLLM and Pylet
    uv pip install -e .
    uv pip install pylet
    ```

3.  Create the worker environment:
    ```bash
    # Create and activate a virtual environment (uv recommended)
    uv venv sllm-worker && source sllm-worker/bin/activate

    # Install sllm_store
    cd sllm_store && rm -rf build
    uv pip install -e .
    cd ..

    # Install ServerlessLLM (worker version)
    uv pip install -e ".[worker]"
    ```

### vLLM Patch

To use vLLM with ServerlessLLM, you must apply a patch. The patch file is located at `sllm_store/vllm_patch/sllm_load.patch` within the ServerlessLLM repository. This patch has been tested with vLLM version `0.11.2`.

Apply the patch using the following script:

```bash
source sllm-worker/bin/activate
./sllm_store/vllm_patch/patch.sh
```

## Running ServerlessLLM Locally

These steps describe how to run ServerlessLLM on your local machine.

### 1. Start a Local Pylet Cluster

First, initiate a local Pylet cluster. This cluster will consist of one head node and one worker node (on the same machine).

**Start the Pylet head node:**

Open a new terminal and run:

```bash
source sllm-head/bin/activate
pylet start
```

You can verify the Pylet head is running:

```bash
curl http://localhost:8000/workers
```

**Start the Pylet worker node:**

Open another new terminal and run:

```bash
source sllm-worker/bin/activate
export CUDA_VISIBLE_DEVICES=0 # Or your desired GPU ID
pylet worker --head http://localhost:8000
```

### 2. Start the ServerlessLLM Store Server

Next, start the ServerlessLLM Store server. By default, it uses `./models` as the storage path.

Open a new terminal and run:

```bash
source sllm-worker/bin/activate
export CUDA_VISIBLE_DEVICES=0 # Or your desired GPU ID
sllm-store start --storage-path ./models --mem-pool-size 4GB
```

Expected output:

```log
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

### 3. Start ServerlessLLM

Now, start the ServerlessLLM service process using `sllm start`.

Open a new terminal and run:

```bash
source sllm-head/bin/activate
sllm start --pylet-endpoint http://localhost:8000
```

At this point, you should have four terminals open: one for the Pylet head node, one for the Pylet worker node, one for the ServerlessLLM Store server, and one for the ServerlessLLM service (started via `sllm start`).

### 4. Deploy a Model

With all services running, you can deploy a model.

Open a new terminal and run:

```bash
source sllm-head/bin/activate
sllm deploy --model facebook/opt-1.3b --backend vllm
```

This command downloads the specified model from Hugging Face Hub. To load a model from a local path, you can use a `config.json` file. Refer to the [CLI API documentation](../api/cli.md#example-configuration-file-configjson) for details.

### 5. Query the Model

Once the model is deployed, you can query it using any OpenAI API-compatible client. For example, use the following `curl` command:

```bash
curl http://127.0.0.1:8343/v1/chat/completions \
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
{"id":"chatcmpl-9f812a40-6b96-4ef9-8584-0b8149892cb9","object":"chat.completion","created":1720021153,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

## Clean Up

To delete a deployed model, use the following command:

```bash
sllm delete facebook/opt-1.3b --backend vllm
```

This command removes the specified model from the ServerlessLLM server.