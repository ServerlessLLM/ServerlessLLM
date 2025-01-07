---
sidebar_position: 1
---

# Installation with ROCm (Experimental)

## Tested Hardware
+ OS: Ubuntu 22.04
+ ROCm: 6.2
+ PyTorch: 2.3.0
+ GPU: MI100s (gfx908), MI200s (gfx90a), Radeon RX 7900 XTX (gfx1100)

## Option 1: Build and run with docker compose (recommended)

### Step 1: Clone the ServerlessLLM Repository

If you haven't already, clone the ServerlessLLM repository:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store/docker/
```

### Step 2:  Configuration

Create a directory on your host machine where models will be stored and set the MODEL_FOLDER environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace /path/to/your/models with the actual path where you want to store the models.

### Step 3: Start the Services

Start the ServerlessLLM services using docker compose:

```bash
docker compose -f docker-compose-amd.yml up -d --build
```

This command will start a container defined in `docker-compose-amd.yml` file. The `sllm-store-server` has already been started in the container with model storage path set to `$STORAGE_PATH`. The `$STORAGE_PATH` is mounted to the host machine's `$MODEL_FOLDER`.

### Step 4: Enter the container and use ServerlessLLM Store

```bash
docker exec -it sllm_store_rocm bash
```

After entering the container, you can use our scripts to save and load models. For example, you can save and load the model in transformers library:

``` bash
python3 examples/sllm_store/save_transformers_model.py --model_name facebook/opt-1.3b --storage_path /models
python3 examples/sllm_store/load_transformers_model.py --model_name facebook/opt-1.3b --storage_path /models
```

Or you can also try to use our integration with vLLM:
:::tip
This function is experimental on ROCm devices. We are currently working on updating the vLLM library to better support ROCm devices.

In the `Dockerfile.rocm`, we have already built the vLLM v0.5.0.post1 from source and applied our patch to the installed vLLM library. If you face any issues, you may change the content to better build the vLLM from source.

For users with gfx1100 (Radeon RDNA3) GPUs, you may need to set the environment variable `VLLM_USE_TRITON_FLASH_ATTN=0` to avoid the issue that vLLM cannot be loaded. This issue is because the flash attention support for AMD GPUs currently does not support gfx1100 GPUs. For more information, you may check this [issue](https://github.com/vllm-project/vllm/issues/4514).
``` bash
export VLLM_USE_TRITON_FLASH_ATTN=0
```
:::

``` bash
python3 examples/sllm_store/save_vllm_model.py --model_name facebook/opt-1.3b --storage_path /models
python3 examples/sllm_store/load_vllm_model.py --model_name facebook/opt-1.3b --storage_path /models
```

## Option 2: Build the wheel from source and install
ServerlessLLM Store (`sllm-store`) currently provides experimental support for ROCm platform. Due to an internal bug in ROCm, serverless-llm-store may face a GPU memory leak in ROCm before version 6.2.0, as noted in [issue](https://github.com/ROCm/HIP/issues/3580).

Currently, `pip install .` does not work with ROCm. We suggest you build `sllm-store` wheel and manually install it in your environment.

To build `sllm-store` from source, we suggest you using the docker and build it in ROCm container.

1. Clone the repository and enter the `store` directory:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
```

2. Build the Docker image from `Dockerfile.builder.rocm`. The `Dockerfile.builder.rocm` is build on top of `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0` image.

``` bash
docker build -t sllm_store_rocm_builder -f Dockerfile.builder.rocm .
```

3. Build the package inside the ROCm docker container
``` bash
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_rocm_builder /bin/bash
rm -rf /app/dist/* # remove the existing built files
python setup.py sdist bdist_wheel
```

4. Install pytorch and package in local environment
``` bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
pip install dist/*.whl
```

## Known issues

1. GPU memory leak in ROCm before version 6.2.0.

This issue is due to an internal bug in ROCm. After the inference instance is completed, the GPU memory is still occupied and not released. For more information, please refer to [issue](https://github.com/ROCm/HIP/issues/3580).

2. vLLM v0.5.0.post1 can not be built in ROCm 6.2.0

This issue is due to the ambiguity of a function call in ROCm 6.2.0. You may change the vLLM's source code as in this [commit](https://github.com/vllm-project/vllm/commit/9984605412de1171a72d955cfcb954725edd4d6f).
