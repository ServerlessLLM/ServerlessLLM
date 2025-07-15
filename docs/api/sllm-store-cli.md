---
sidebar_position: 2
---

# ServerlessLLM Store CLI

ServerlessLLM Store's CLI allows the use `sllm-store`'s functionalities within a terminal window. It has the functions:
- `start`: Starts the gRPC server with the specified configuration.
- `save`: Convert a HuggingFace model into a loading-optimized format and save it to a local path.
- `load`: Load a model into given GPUs.

## Requirements
- OS: Ubuntu 22.04
- Python: 3.10
- GPU: compute capability 7.0 or higher

## Installations

### Create a virtual environment
``` bash
conda create -n sllm-store python=3.10 -y
conda activate sllm-store
```

### Install C++ Runtime Library (required for compiling and running CUDA/C++ extensions)
``` bash
conda install -c conda-forge libstdcxx-ng=12 -y
```

### Install with pip
```bash
pip install serverless-llm-store
```

## Example Workflow
1. Firstly, start the ServerlessLLM Store server. By default, it uses ./models as the storage path.
Launch the checkpoint store server in a separate process:
``` bash
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.
sllm-store start --storage-path $PWD/models --mem-pool-size 4GB
```

2. Convert a model to ServerlessLLM format and save it to a local path:
``` bash
sllm-store save --model facebook/opt-1.3b --backend vllm
```

3. Load a previously saved model into memory, ready for inference:
```bash
sllm-store load --model facebook/opt-1.3b --backend vllm
```

## sllm-store start

Start a gRPC server to serve models stored using ServerlessLLM. This enables fast, low-latency access to models registered via sllm-store save, allowing external clients to load model weights, retrieve metadata, and perform inference-related operations efficiently.

The server supports in-memory caching with customizable memory pooling and chunking, optimized for parallel read access and minimal I/O latency.

#### Usage
```bash
sllm-store start [OPTIONS]
```

#### Options

- `--host <host>`
  - Host address to bind the gRPC server to.

- `--port <port>`
  - Port number on which the gRPC server will listen for incoming requests.

- `--storage-path <storage_path>`
  - Path to the directory containing models previously saved with sllm-store save.

- `--num-thread <num_thread>`
  - Number of threads to use for I/O operations and chunk handling.

- `--chunk-size <chunk_size>`
  - Size of individual memory chunks used for caching model data (e.g., 64MiB, 512KB). Must include unit suffix.

- `--mem-pool-size <mem_pool_size>`
  - Total memory pool size to allocate for the in-memory cache (e.g., 4GiB, 2GB). Must include unit suffix.

- `--disk-size <disk_size>`
  - (Currently unused) Would set the maximum size sllm-store can occupy in disk cache.

- `--registration-required`
  - If specified, models must be registered with the server before loading.

#### Examples

Start the server using all default values:
``` bash
sllm-store start
```

Start the server with a custom storage path:
``` bash
sllm-store start --storage-path /your/folder
```

Specify a custom port and host:
``` bash
sllm-store start --host 127.0.0.1 --port 9090
```

Use larger chunk size and memory pool for large models in a multi-threaded environment:
``` bash
sllm-store start --num-thread 16 --chunk-size 128MB --mem-pool-size 8GB
```

Run with access control enabled:
``` bash
sllm-store start --registration-required True
```

Full example for production-style setup:
``` bash
sllm-store start \
  --host 0.0.0.0 \
  --port 8000 \
  --storage-path /data/models \
  --num-thread 8 \
  --chunk-size 64MB \
  --mem-pool-size 16GB \
  --registration-required True
```

## sllm-store save

Saves a model to a local directory through a backend of choice, making it available for future inference requests. Only model name and backend are required, with the rest having default values.

It supports download of [PEFT LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/peft/main/en/index) for transformer models, and varying tensor sizes for parallel download of vLLM models.


#### Usage
```bash
sllm-store save [OPTIONS]
```

#### Options

- `--model <model_name>`
  - Model name to deploy with default configuration. The model name must be a Hugging Face pretrained model name. You can find the list of available models [here](https://huggingface.co/models).

- `--backend <backend_name>`
  - Select a backend for the model to be converted to `ServerlessLLM format` from. Supported backends are `vllm` and `transformers`.

- `--adapter`
  - Enable LoRA adapter support. Overwrite `adapter`, which is by default set to False. Only `transformers` backend is supported.

- `--adapter-name <adapter_name>`
  - Adapter name to save. Must be a Hugging Face pretrained LoRA adapter name.

- `--tensor-parallel-size <tensor_parallel_size>`
  - Number of GPUs you want to use. Only `vllm` backend is supported.

- `--local-model-path <local_model_path>`
  - Saves the model from a local path if it contains a Hugging Face snapshot of the model.

- `--storage-path <storage_path>`
  - Location where the model will be saved.

#### Examples
Save a vLLM model name with default configuration:
```bash
sllm-store save --model facebook/opt-1.3b --backend vllm
```

Save a transformers model to a set location:
```bash
sllm-store save --model facebook/opt-1.3b --backend vllm --storage-path ./your/folder
```

Save a vLLM model from a locally stored snapshot and overwrite the tensor parallel size:
```bash
sllm-store save --model facebook/opt-1.3b --backend vllm --tensor-parallel-size 4 --local-model-path ./path/to/snapshot
```

Save a transformers model with a LoRA adapter:
```bash
sllm-store save --model facebook/opt-1.3b --backend transformers --adapter --adapter-name crumb/FLAN-OPT-1.3b-LoRA
```

## sllm-store load

Load a model from local storage and run example inference to verify deployment. This command supports both the transformers and vllm backends, with optional support for PEFT LoRA adapters and quantized precision formats including int8, fp4, and nf4 (LoRA and quantization supported on transformers backend only).

When using the transformers backend, the function warms up GPU devices, loads the base model from disk, and optionally merges a LoRA adapter if specified. With vllm, it loads the model in the ServerlessLLM format.

#### Usage
```bash
sllm-store load [OPTIONS]
```

#### Options

- `--model <model_name>`
  - Model name to deploy with default configuration. The model name must be a Hugging Face pretrained model name. You can find the list of available models [here](https://huggingface.co/models).

- `--backend <backend_name>`
  - Select a backend for the model to be converted to `ServerlessLLM format` from. Supported backends are `vllm` and `transformers`.

- `--adapter`
  - Enable LoRA adapter support for the transformers backend. Overwrite `adapter` in the default configuration (`transformers` backend only).

- `--adapter-name <adapter_name>`
  - Adapter name to save. Must be a Hugging Face pretrained LoRA adapter name.

- `--precision <precision>`
  - Precision to use when loading the model (`transformers` backend only). For more info on quantization in ServerlessLLM, visit [here](https://serverlessllm.github.io/docs/stable/store/quickstart#quantization).

- `--storage-path <storage_path>`
  - Location where the model will be loaded from.

#### Examples
Load a vllm model from storage:
``` bash
sllm-store load --model facebook/opt-1.3b --backend vllm
```

Load a transformers model from storage with int8 quantization:
``` bash
sllm-store load --model facebook/opt-1.3b --backend transformers --precision int8 --storage-path ./your/models
```

Load a transformers model with a LoRA adapter:
``` bash
sllm-store load --model facebook/opt-1.3b --backend transformers --adapter --adapter-name crumb/FLAN-OPT-1.3b-LoRA
```

#### Note: loading vLLM models

To load models with vLLM, you need to apply a compatibility patch to your vLLM installation. This patch has been tested with vLLM version `0.9.0.1`.

```bash
   ./sllm_store/vllm_patch/patch.sh
```

:::note
The patch file is located at `sllm_store/vllm_patch/sllm_load.patch` in the ServerlessLLM repository.
:::