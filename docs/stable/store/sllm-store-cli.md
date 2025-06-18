---
sidebar_position: 2
---

# ServerlessLLM Store CLI

ServerlessLLM Store's CLI allows the use `sllm-store`'s functionalities within a terminal window. It has the functions:
- `save`: Convert a HuggingFace model into a loading-optimized format and save it to a local path.
- `load`: Load a model into given GPUs.


## Requirements
- OS: Ubuntu 20.04
- Python: 3.10
- GPU: compute capability 7.0 or higher

## Installations

### Create a virtual environment
```bash
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

### Install from source
1. Clone the repository and enter the `store` directory

``` bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
```

2. Install the package from source

```bash
rm -rf build
pip install .
```

## Getting Started
### 1. Start the ServerlessLLM Store Server
Firstly, start the ServerlessLLM Store server. By default, it uses ./models as the storage path.

Launch the checkpoint store server in a separate process:
``` bash
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.
sllm-store start --storage-path $PWD/models --mem-pool-size 4GB
```

### 2. vLLM Patch (optional)

To use vLLM with ServerlessLLM, you must apply a patch. This patch has been tested with vLLM version 0.9.0.1.

1. **Check patch status** (optional):
   ```bash
   ./sllm_store/vllm_patch/check_patch.sh
   ```

2. **Apply the patch**:
   ```bash
   ./sllm_store/vllm_patch/patch.sh
   ```

3. **Remove the patch** (if needed):
   ```bash
   ./sllm_store/vllm_patch/remove_patch.sh
   ```

:::note
The patch file is located at `sllm_store/vllm_patch/sllm_load.patch` in the ServerlessLLM repository.
:::

## Example Workflow
1. Convert a model to ServerlessLLM format and save it to a local path:
``` bash
sllm-store save --model facebook/opt-1.3b --backend vllm
```

2. Load a previously saved model into memory, ready for inference:
```bash
sllm-store load --model facebook/opt-1.3b --backend vllm
```

## sllm-store save

Save a model to a storage backend using a unique identifier, making it available for future inference requests. This function supports both in-memory and persistent storage options, depending on system configuration.

Designed for efficiency and reusability, save allows models to be registered once and shared across multiple inference sessions without repeated setup overhead.

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
  - Enable LoRA adapter support for the `transformers` backend. Overwrite `adapter` in the default configuration.

- `--adapter-name <adapter_name>`
  - Adapter name to save. Must be a Hugging Face pretrained LoRA adapter name.

- `--tensor-parallel-size <tensor_parallel_size>`
  - Number of GPUs you want to use.

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
sllm-cli deploy --model facebook/opt-1.3b --backend transformers --adapter --adapter-name crumb/FLAN-OPT-1.3b-LoRA
```

## sllm-store load

Load a model from local storage and run example inference to verify deployment. This command supports both the transformers and vllm backends, with optional support for PEFT LoRA adapters and quantized precision formats including int8, fp4, and nf4.

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
