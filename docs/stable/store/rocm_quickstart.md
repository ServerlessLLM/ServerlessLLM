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

Try to save and load a model in vLLM:

``` bash
python3 examples/save_vllm_model.py --model-name "facebook/opt-125m" --storage-path "/models"
python3 examples/load_vllm_model.py --model-name "facebook/opt-125m" --storage-path "/models"
```
Expected output:

``` bash
WARNING 03-13 09:37:29 rocm.py:31] `fork` method is not supported by ROCm. VLLM_WORKER_MULTIPROC_METHOD is overridden to `spawn` instead.
INFO 03-13 09:37:35 config.py:510] This model supports multiple tasks: {'embed', 'classify', 'generate', 'reward', 'score'}. Defaulting to 'generate'.
INFO 03-13 09:37:35 config.py:1339] Disabled the custom all-reduce kernel because it is not supported on AMD GPUs.
INFO 03-13 09:37:35 llm_engine.py:234] Initializing an LLM engine (v0.6.6) with config: model='/models/facebook/opt-125m', speculative_config=None, tokenizer='/models/facebook/opt-125m', skip_tokenizer_init=False,
 tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=serverless_llm, tensor_para
llel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(
guided_decoding_backend='xgrammar'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=/models/faceb
ook/opt-125m, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=False, chunked_prefill_enabled=False, use_async_output_proc=True, disable_mm_preprocessor_cache=False, mm_processor_kwargs
=None, pooler_config=None, compilation_config={"splitting_ops":["vllm.unified_attention","vllm.unified_attention_with_output"],"candidate_compile_sizes":[],"compile_sizes":[],"capture_sizes":[256,248,240,232,224,2
16,208,200,192,184,176,168,160,152,144,136,128,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,4,2,1],"max_capture_size":256}, use_cached_outputs=False,
INFO 03-13 09:37:38 selector.py:134] Using ROCmFlashAttention backend.
INFO 03-13 09:37:39 model_runner.py:1094] Starting to load model /models/facebook/opt-125m...
DEBUG 03-13 09:37:39 torch.py:137] allocate_cuda_memory takes 0.0004572868347167969 seconds
DEBUG 03-13 09:37:39 client.py:72] load_into_gpu: facebook/opt-125m/rank_0, 8554547c-25d3-4a01-92b6-27d69d91d3b8
INFO 03-13 09:37:39 client.py:113] Model loaded: facebook/opt-125m/rank_0, 8554547c-25d3-4a01-92b6-27d69d91d3b8
INFO 03-13 09:37:39 torch.py:160] restore state_dict takes 0.00017452239990234375 seconds
INFO 03-13 09:37:39 client.py:117] confirm_model_loaded: facebook/opt-125m/rank_0, 8554547c-25d3-4a01-92b6-27d69d91d3b8
INFO 03-13 09:37:39 client.py:125] Model loaded
INFO 03-13 09:37:39 model_runner.py:1099] Loading model weights took 0.0000 GB
/app/third_party/vllm/vllm/model_executor/layers/linear.py:140: UserWarning: Attempting to use hipBLASLt on an unsupported architecture! Overriding blas backend to hipblas (Triggered internally at ../aten/src/ATen
/Context.cpp:296.)
  return F.linear(x, layer.weight, bias)
INFO 03-13 09:37:42 worker.py:253] Memory profiling takes 2.68 seconds
INFO 03-13 09:37:42 worker.py:253] the current vLLM instance can use total_gpu_memory (23.98GiB) x gpu_memory_utilization (0.90) = 21.59GiB
INFO 03-13 09:37:42 worker.py:253] model weights take 0.00GiB; non_torch_memory takes 0.62GiB; PyTorch activation peak memory takes 0.46GiB; the rest of the memory reserved for KV Cache is 20.50GiB.
INFO 03-13 09:37:42 gpu_executor.py:76] # GPU blocks: 37326, # CPU blocks: 7281
INFO 03-13 09:37:42 gpu_executor.py:80] Maximum concurrency for 2048 tokens per request: 291.61x
INFO 03-13 09:37:43 model_runner.py:1429] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--
enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decre
ase memory usage.
Capturing CUDA graph shapes: 100%|████████████████████████████████████████| 35/35 [00:09<00:00,  3.73it/s]
INFO 03-13 09:37:52 model_runner.py:1549] Graph capturing finished in 9 secs, took 0.06 GiB
INFO 03-13 09:37:52 llm_engine.py:431] init engine (profile, create kv cache, warmup model) took 12.80 seconds
Processed prompts: 100%|█| 4/4 [00:00<00:00, 50.16it/s, est. speed input: 326.19 toks/s, output: 802.89 to
Prompt: 'Hello, my name is', Generated text: ' Joel, my dad is my friend and we are in a relationship. I am'
Prompt: 'The president of the United States is', Generated text: ' speaking out against the release of some State Department documents which show the Russians were involved'
Prompt: 'The capital of France is', Generated text: ' a worldwide knowledge center. What better place to learn about the history and culture of'
Prompt: 'The future of AI is', Generated text: " here: it's the future of everything\nIf you want to test your minds"
[rank0]:[W313 09:37:53.050846849 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_p
rocess_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This cons
traint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())

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
