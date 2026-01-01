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

We provide a Dockerfile with ROCm support. Currently, it's built on base image `rocm/vllm-dev:base_ROCm-6.3.1_20250528_tuned_20250530`

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
INFO 06-05 12:59:07 cli.py:76] Starting gRPC server
INFO 06-05 12:59:07 server.py:40] StorageServicer: storage_path=/models, mem_pool_size=4294967296, num_thread=4, chunk_size=33554432, registration_required=False
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250605 12:59:11.141070     1 checkpoint_store_hip.cpp:42] Number of GPUs: 1
I20250605 12:59:11.141098     1 checkpoint_store_hip.cpp:44] I/O threads: 4, chunk size: 32MB
I20250605 12:59:11.141103     1 checkpoint_store_hip.cpp:46] Storage path: "/models"
I20250605 12:59:11.141119     1 checkpoint_store_hip.cpp:72] GPU 0 UUID: 61363865-3865-3038-3831-366132376261
I20250605 12:59:11.519277     1 pinned_memory_pool_hip.cpp:30] Creating PinnedMemoryPool with 128 buffers of 33554432 bytes
I20250605 12:59:12.487957     1 checkpoint_store_hip.cpp:84] Memory pool created with 4GB
INFO 06-05 12:59:12 server.py:231] Starting gRPC server on 0.0.0.0:8073

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
DEBUG 06-05 13:01:01 transformers.py:203] load_dict_non_blocking takes 0.0071375370025634766 seconds
DEBUG 06-05 13:01:01 transformers.py:213] load config takes 0.003943443298339844 seconds
DEBUG 06-05 13:01:01 torch.py:137] allocate_cuda_memory takes 0.0012660026550292969 seconds
DEBUG 06-05 13:01:01 client.py:72] load_into_gpu: facebook/opt-1.3b, 93b1932e-4b43-42cb-b82d-7228ef21810b
INFO 06-05 13:01:01 client.py:113] Model loaded: facebook/opt-1.3b, 93b1932e-4b43-42cb-b82d-7228ef21810b
INFO 06-05 13:01:01 torch.py:160] restore state_dict takes 0.0004298686981201172 seconds
DEBUG 06-05 13:01:02 transformers.py:224] load model takes 0.9706132411956787 seconds
INFO 06-05 13:01:02 client.py:117] confirm_model_loaded: facebook/opt-1.3b, 93b1932e-4b43-42cb-b82d-7228ef21810b
INFO 06-05 13:01:06 client.py:125] Model loaded
Model loading time: 5.28s
tokenizer_config.json: 100%|██████████████████████████████| 685/685 [00:00<00:00, 6.68MB/s]
vocab.json: 100%|███████████████████████████████████████| 899k/899k [00:00<00:00, 4.05MB/s]
merges.txt: 100%|███████████████████████████████████████| 456k/456k [00:00<00:00, 3.05MB/s]
special_tokens_map.json: 100%|████████████████████████████| 441/441 [00:00<00:00, 4.10MB/s]
/usr/local/lib/python3.12/dist-packages/torch/nn/modules/linear.py:125: UserWarning: Failed validator: GCN_ARCH_NAME (Triggered internally at /app/pytorch/aten/src/ATen/hip/tunable/Tunable.cpp:366.)
  return F.linear(input, self.weight, self.bias)
Hello, my dog is cute and I want to give him a good home. I have a lot of experience with dogs and I
```

Try to save and load a model in vLLM:

``` bash
python3 examples/save_vllm_model.py --model-name "facebook/opt-125m" --storage-path "/models"
python3 examples/load_vllm_model.py --model-name "facebook/opt-125m" --storage-path "/models"
```
Expected output:

``` bash
INFO 06-05 13:02:51 [__init__.py:243] Automatically detected platform rocm.
INFO 06-05 13:02:52 [__init__.py:31] Available plugins for group vllm.general_plugins:
INFO 06-05 13:02:52 [__init__.py:33] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
INFO 06-05 13:02:52 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-05 13:03:00 [config.py:793] This model supports multiple tasks: {'reward', 'embed', 'generate', 'classify', 'score'}. Defaulting to 'generate'.
INFO 06-05 13:03:00 [arg_utils.py:1594] rocm is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.
INFO 06-05 13:03:04 [config.py:1910] Disabled the custom all-reduce kernel because it is not supported on current platform.
INFO 06-05 13:03:04 [llm_engine.py:230] Initializing a V0 LLM engine (v0.11.2) with config: model='/models/facebook/opt-125m', speculative_config=None, tokenizer='/models/facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.SERVERLESS_LLM, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/models/facebook/opt-125m, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"compile_sizes": [], "inductor_compile_config": {"enable_auto_functionalized_v2": false}, "cudagraph_capture_sizes": [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1], "max_capture_size": 256}, use_cached_outputs=False,
INFO 06-05 13:03:04 [rocm.py:208] None is not supported in AMD GPUs.
INFO 06-05 13:03:04 [rocm.py:209] Using ROCmFlashAttention backend.
INFO 06-05 13:03:05 [parallel_state.py:1064] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
INFO 06-05 13:03:05 [model_runner.py:1170] Starting to load model /models/facebook/opt-125m...
DEBUG 06-05 13:03:05 torch.py:137] allocate_cuda_memory takes 0.0004763603210449219 seconds
DEBUG 06-05 13:03:05 client.py:72] load_into_gpu: facebook/opt-125m/rank_0, e8e7d900-652d-4822-8992-ad22f734b9c8
INFO 06-05 13:03:05 client.py:113] Model loaded: facebook/opt-125m/rank_0, e8e7d900-652d-4822-8992-ad22f734b9c8
INFO 06-05 13:03:05 torch.py:160] restore state_dict takes 0.00021338462829589844 seconds
INFO 06-05 13:03:05 client.py:117] confirm_model_loaded: facebook/opt-125m/rank_0, e8e7d900-652d-4822-8992-ad22f734b9c8
INFO 06-05 13:03:05 client.py:125] Model loaded
INFO 06-05 13:03:05 [model_runner.py:1202] Model loading took 0.2363 GiB and 0.711783 seconds
/app/third_party/vllm/vllm/model_executor/layers/utils.py:80: UserWarning: Failed validator: GCN_ARCH_NAME (Triggered internally at /app/pytorch/aten/src/ATen/hip/tunable/Tunable.cpp:366.)
  return torch.nn.functional.linear(x, weight, bias)
INFO 06-05 13:03:17 [worker.py:303] Memory profiling takes 11.68 seconds
INFO 06-05 13:03:17 [worker.py:303] the current vLLM instance can use total_gpu_memory (23.98GiB) x gpu_memory_utilization (0.90) = 21.59GiB
INFO 06-05 13:03:17 [worker.py:303] model weights take 0.24GiB; non_torch_memory takes 0.53GiB; PyTorch activation peak memory takes 0.49GiB; the rest of the memory reserved for KV Cache is 20.33GiB.
INFO 06-05 13:03:17 [executor_base.py:112] # rocm blocks: 37011, # CPU blocks: 7281
INFO 06-05 13:03:17 [executor_base.py:117] Maximum concurrency for 2048 tokens per request: 289.15x
INFO 06-05 13:03:18 [model_runner.py:1526] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI. If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
Capturing CUDA graph shapes: 100%|█████████████████████████| 35/35 [00:09<00:00,  3.55it/s]
INFO 06-05 13:03:28 [model_runner.py:1684] Graph capturing finished in 10 secs, took 0.13 GiB
INFO 06-05 13:03:28 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 22.81 seconds
Adding requests: 100%|█████████████████████████████████████| 4/4 [00:00<00:00, 2079.22it/s]
Processed prompts: 100%|█| 4/4 [00:00<00:00,  6.71it/s, est. speed input: 43.59 toks/s, out
Prompt: 'Hello, my name is', Generated text: ' Joel, my dad is my friend and we are in a relationship. I am'
Prompt: 'The president of the United States is', Generated text: ' speaking out against the release of some State Department documents which show the Russians were involved'
Prompt: 'The capital of France is', Generated text: ' a worldwide knowledge center. What better place to learn about the history and culture of'
Prompt: 'The future of AI is', Generated text: " here: it's the future of everything\nIf you want to test your minds"
[rank0]:[W605 13:03:30.532018298 ProcessGroupNCCL.cpp:1476] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
```

## Build the wheel from source and install

Currently, `pip install .` does not work with ROCm. We suggest you build `sllm-store` wheel and manually install it in your environment.



1. If there's a customized PyTorch version installed, you may need to run the following command to modify the `torch` version in `requirements.txt`:

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

