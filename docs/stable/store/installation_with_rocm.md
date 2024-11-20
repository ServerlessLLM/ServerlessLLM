---
sidebar_position: 1
---

# Installation with ROCm (Experimental)

## Latest Tested Version
+ v0.5.1

## Tested Hardware
+ OS: Ubuntu 22.04
+ ROCm: 6.2
+ PyTorch: 2.3.0
+ GPU: MI100s (gfx908), MI200s (gfx90a)

## Build the wheel from source and install
ServerlessLLM Store (`sllm-store`) currently provides experimental support for ROCm platform. Due to an internal bug in ROCm, serverless-llm-store may face a GPU memory leak in ROCm before version 6.2.0, as noted in [issue](https://github.com/ROCm/HIP/issues/3580).

Currently, `pip install .` does not work with ROCm. We suggest you build `sllm-store` wheel and manually install it in your environment.

To build `sllm-store` from source, we suggest you using the docker and build it in ROCm container.

1. Clone the repository and enter the `store` directory:

```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
```

2. Build the Docker image from `Dockerfile.rocm`. The `Dockerfile.rocm` is build on top of `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0` image.

``` bash
docker build -t sllm_store_rocm -f Dockerfile.rocm .
```

3. Build the package inside the ROCm docker container
``` bash
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_rocm /bin/bash
rm -rf /app/dist/* # remove the existing built files
python setup.py sdist bdist_wheel
```

4. Install pytorch and package in local environment
``` bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
pip install dist/*.whl
```

## Verify the Installation

### End to end tests

#### Transformer model Loading and Inference

1. Save the `facebook/opt-1.3b` model in `./models` directory

``` bash
python3 examples/sllm_store/save_transformers_model.py --model_name facebook/opt-1.3b --storage_path ./models
```

2. Start the `sllm-store` server

``` bash
sllm-store-server
```

3. Load the model and run the inference in another terminal

``` bash
python3 examples/sllm_store/load_transformers_model.py --model_name facebook/opt-1.3b --storage_path ./models
```

Expected Output:

``` bash
DEBUG 10-31 10:43:14 transformers.py:178] load_dict_non_blocking takes 0.008747100830078125 seconds
DEBUG 10-31 10:43:14 transformers.py:189] load config takes 0.0016036033630371094 seconds
DEBUG 10-31 10:43:14 torch.py:137] allocate_cuda_memory takes 0.0041697025299072266 seconds
DEBUG 10-31 10:43:14 client.py:72] load_into_gpu: facebook/opt-1.3b, 544e032d-9080-429f-bbc0-cdbc2a298060
INFO 10-31 10:43:14 client.py:113] Model loaded: facebook/opt-1.3b, 544e032d-9080-429f-bbc0-cdbc2a298060
INFO 10-31 10:43:14 torch.py:160] restore state_dict takes 0.0017423629760742188 seconds
DEBUG 10-31 10:43:14 transformers.py:199] load model takes 0.17534756660461426 seconds
INFO 10-31 10:43:14 client.py:117] confirm_model_loaded: facebook/opt-1.3b, 544e032d-9080-429f-bbc0-cdbc2a298060
INFO 10-31 10:43:14 client.py:125] Model loaded
Model loading time: 0.20s
~/miniconda3/envs/sllm/lib/python3.10/site-packages/transformers/generation/utils.py:1249: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
Hello, my dog is cute and I want to give him a good home. I have a
```

#### vLLM model Loading and Inference
:::tip
Directly installing vLLM v0.5.0.post1 may not work with ROCm 6.2.0. This issue is due to the ambiguity of a function call in ROCm 6.2.0. You may change the vLLM's source code as in this [commit](https://github.com/vllm-project/vllm/commit/9984605412de1171a72d955cfcb954725edd4d6f).

Similar as in CUDA, you need to apply our patch `sllm_store/vllm_patch/sllm_load.patch` to the installed vLLM library.
```bash
./sllm_store/vllm_patch/patch.sh
```
:::

1. Save the `facebook/opt-1.3b` model in `./models` directory

``` bash
python3 examples/sllm_store/save_vllm_model.py --model_name facebook/opt-1.3b --storage_path ./models
```

2. Start the `sllm-store` server

``` bash
sllm-store-server
```

3. Load the model and run the inference in another terminal

``` bash
python3 examples/sllm_store/load_vllm_model.py --model_name facebook/opt-1.3b --storage_path ./models
```

Expected Output:

``` bash
INFO 10-31 11:05:16 llm_engine.py:161] Initializing an LLM engine (v0.5.0) with config: model='./models/facebook/opt-1.3b', speculative_config=None, tokenizer='./models/facebook/opt-1.3b', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.SERVERLESS_LLM, tensor_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), seed=0, served_model_name=./models/facebook/opt-1.3b)
INFO 10-31 11:05:17 selector.py:56] Using ROCmFlashAttention backend.
INFO 10-31 11:05:17 selector.py:56] Using ROCmFlashAttention backend.
DEBUG 10-31 11:05:17 torch.py:137] allocate_cuda_memory takes 0.0005428791046142578 seconds
DEBUG 10-31 11:05:17 client.py:72] load_into_gpu: facebook/opt-1.3b/rank_0, 9d7c0425-f652-4c4c-b1c5-fb6df0aab0a8
INFO 10-31 11:05:17 client.py:113] Model loaded: facebook/opt-1.3b/rank_0, 9d7c0425-f652-4c4c-b1c5-fb6df0aab0a8
INFO 10-31 11:05:17 torch.py:160] restore state_dict takes 0.0013034343719482422 seconds
INFO 10-31 11:05:17 client.py:117] confirm_model_loaded: facebook/opt-1.3b/rank_0, 9d7c0425-f652-4c4c-b1c5-fb6df0aab0a8
INFO 10-31 11:05:17 client.py:125] Model loaded
INFO 10-31 11:05:17 model_runner.py:160] Loading model weights took 0.0000 GB
INFO 10-31 11:05:25 gpu_executor.py:83] # GPU blocks: 18509, # CPU blocks: 1365
INFO 10-31 11:05:26 model_runner.py:903] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 10-31 11:05:26 model_runner.py:907] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 10-31 11:05:31 model_runner.py:979] Graph capturing finished in 6 secs.
Processed prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 12.13it/s, est. speed input: 78.83 toks/s, output: 194.04 toks/s]
Prompt: 'Hello, my name is', Generated text: ' Joel, and I have been working as a web designer/developer for the'
Prompt: 'The president of the United States is', Generated text: " speaking in an increasingly important national security forum and he's not using the right words"
Prompt: 'The capital of France is', Generated text: " Paris.\nYeah but you couldn't get it through a French newspaper!"
Prompt: 'The future of AI is', Generated text: ' literally in your hands\nDespite all the hype, AI isn’t here'
```

### Python tests

1. Install the test dependencies

```bash
cd ServerlessLLM
pip install -r requirements-test.txt
```

2. Run the tests
```
cd ServerlessLLM/sllm_store/tests/python
pytest
```

### C++ tests

1. Build the C++ tests

```bash
cd ServerlessLLM/sllm_store
bash build.sh
```

2. Run the tests

```bash
cd ServerlessLLM/sllm_store/build
ctest --output-on-failure
```

## Known issues

1. GPU memory leak in ROCm before version 6.2.0.

This issue is due to an internal bug in ROCm. After the inference instance is completed, the GPU memory is still occupied and not released. For more information, please refer to [issue](https://github.com/ROCm/HIP/issues/3580).

2. vLLM v0.5.0.post1 can not be built in ROCm 6.2.0

This issue is due to the ambiguity of a function call in ROCm 6.2.0. You may change the vLLM's source code as in this [commit](https://github.com/vllm-project/vllm/commit/9984605412de1171a72d955cfcb954725edd4d6f).
