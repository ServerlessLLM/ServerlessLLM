---
sidebar_position: 1
---

# ROCm Quick Start

ServerlessLLM Store (`sllm-store`) currently supports ROCm platform. However, there are no pre-built wheels for ROCm.

Due to an internal bug in ROCm, serverless-llm-store may face a GPU memory leak in ROCm before version 6.2.0, as noted in [issue](https://github.com/ROCm/HIP/issues/3580).

1. Clone the repository and enter the `store` directory:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
```
After that, you may either use the Docker image or build the `sllm-store` wheel from source and install it in your environment.

## Use the Docker image

We provide a Docker file with ROCm support. Currently, it's built on base image `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0`

2. Build the Docker image:

``` bash
docker build -t sllm_store_rocm -f Dockerfile.rocm .
```

3. Start the Docker container:

:::tip
If you want to run inference outside the Docker container, you need to pass the port to the host machine. For example, `-p 8073:8073`. You can also get the wheel from the Docker container after starting it via `docker cp sllm_store_server:/app/dist .`.
:::

``` bash
docker run --name sllm_store_server --rm -it \
  --device /dev/kfd --device /dev/dri \
  --security-opt seccomp=unconfined \
  -v $(pwd)/models:/models \
  sllm_store_rocm
```

Expected output:

``` bash
INFO 02-13 04:52:36 cli.py:76] Starting gRPC server
INFO 02-13 04:52:36 server.py:40] StorageServicer: storage_path=/models, mem_pool_size=4294967296, num_thread=4, chunk_size=33554432, registration_required=False
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250213 04:52:36.284631     1 checkpoint_store_hip.cpp:42] Number of GPUs: 1
I20250213 04:52:36.284652     1 checkpoint_store_hip.cpp:44] I/O threads: 4, chunk size: 32MB
I20250213 04:52:36.284659     1 checkpoint_store_hip.cpp:46] Storage path: "/models"
I20250213 04:52:36.284674     1 checkpoint_store_hip.cpp:72] GPU 0 UUID: 61363865-3865-3038-3831-366132376261
I20250213 04:52:36.425267     1 pinned_memory_pool_hip.cpp:30] Creating PinnedMemoryPool with 128 buffers of 33554432 bytes
I20250213 04:52:37.333868     1 checkpoint_store_hip.cpp:84] Memory pool created with 4GB
INFO 02-13 04:52:37 server.py:231] Starting gRPC server on 0.0.0.0:8073

```

After starting the Docker container, you can enter the container and run the following command to test the installation.

``` bash
docker exec -it sllm_store_server /bin/bash
```

Try to save and load a transformer model:

``` bash
python3 examples/save_transformers_model.py --model-name "facebook/opt-1.3b" --storage-path "/models"
python3 examples/load_transformers_model.py --model-name "facebook/opt-1.3b" --storage-path "/models"
```
Expected output:

``` bash
DEBUG 02-13 04:58:09 transformers.py:178] load_dict_non_blocking takes 0.005706787109375 seconds
DEBUG 02-13 04:58:09 transformers.py:189] load config takes 0.0013949871063232422 seconds
DEBUG 02-13 04:58:09 torch.py:137] allocate_cuda_memory takes 0.001325368881225586 seconds
DEBUG 02-13 04:58:09 client.py:72] load_into_gpu: facebook/opt-1.3b, d34e8994-37da-4357-a86c-2205175e3b3f
INFO 02-13 04:58:09 client.py:113] Model loaded: facebook/opt-1.3b, d34e8994-37da-4357-a86c-2205175e3b3f
INFO 02-13 04:58:09 torch.py:160] restore state_dict takes 0.0004620552062988281 seconds
DEBUG 02-13 04:58:09 transformers.py:199] load model takes 0.06779956817626953 seconds
INFO 02-13 04:58:09 client.py:117] confirm_model_loaded: facebook/opt-1.3b, d34e8994-37da-4357-a86c-2205175e3b3f
INFO 02-13 04:58:14 client.py:125] Model loaded
Model loading time: 5.14s
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 685/685 [00:00<00:00, 8.26MB/s]
vocab.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 899k/899k [00:00<00:00, 4.05MB/s]
merges.txt: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 3.07MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 441/441 [00:00<00:00, 4.59MB/s]
/opt/conda/envs/py_3.10/lib/python3.10/site-packages/transformers/generation/utils.py:1249: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `
max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
Hello, my dog is cute and I want to give him a good home. I have a

```

## Build the wheel from source and install

Currently, `pip install .` does not work with ROCm. We suggest you build `sllm-store` wheel and manually install it in your environment.



If there's a customized PyTorch version installed, you may need to run the following command to modify the `torch` version in `requirements.txt`:

```bash
python3 using_existing_torch.py
```

2. Build the wheel:

```bash
python setup.py sdist bdist_wheel
```

## Known issues

1. GPU memory leak in ROCm before version 6.2.0.

This issue is due to an internal bug in ROCm. After the inference instance is completed, the GPU memory is still occupied and not released. For more information, please refer to [issue](https://github.com/ROCm/HIP/issues/3580).

2. vLLM v0.5.0.post1 can not be built in ROCm 6.2.0

This issue is due to the ambiguity of a function call in ROCm 6.2.0. You may change the vLLM's source code as in this [commit](https://github.com/vllm-project/vllm/commit/9984605412de1171a72d955cfcb954725edd4d6f).
