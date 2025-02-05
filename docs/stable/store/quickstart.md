---
sidebar_position: 0
---

# Quickstart Guide

ServerlessLLM Store (`sllm-store`) is a Python library that supports fast model checkpoint loading from multi-tier storage (i.e., DRAM, SSD, HDD) into GPUs.

ServerlessLLM Store provides a model manager and two key functions:
- `save_model`: Convert a HuggingFace model into a loading-optimized format and save it to a local path.
- `load_model`: Load a model into given GPUs.


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

## Usage Examples
:::tip
We highly recommend using a fast storage device (e.g., NVMe SSD) to store the model files for the best experience.
For example, create a directory `models` on the NVMe SSD and link it to the local path.
```bash
mkdir -p /mnt/nvme/models   # Replace '/mnt/nvme' with your NVMe SSD path.
ln -s /mnt/nvme/models ./models
```
:::

1. Convert a model to ServerlessLLM format and save it to a local path:
```python
from sllm_store.transformers import save_model

# Load a model from HuggingFace model hub.
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16)

# Replace './models' with your local path.
save_model(model, './models/facebook/opt-1.3b')
```

2. Launch the checkpoint store server in a separate process:
```bash
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.
sllm-store start --storage-path $PWD/models --mem-pool-size 4GB
```

<!-- Running the server using a container:

```bash
docker build -t checkpoint_store_server -f Dockerfile .
# Make sure the models have been downloaded using examples/save_model.py script
docker run -it --rm -v $PWD/models:/app/models checkpoint_store_server
``` -->

3. Load model in your project and make inference:
```python
import time
import torch
from sllm_store.transformers import load_model

# warm up the GPU
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    torch.ones(1).to(f"cuda:{i}")
    torch.cuda.synchronize()

start = time.time()
model = load_model("facebook/opt-1.3b", device_map="auto", torch_dtype=torch.float16, storage_path="./models/", fully_parallel=True)
# Please note the loading time depends on the model size and the hardware bandwidth.
print(f"Model loading time: {time.time() - start:.2f}s")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
inputs = tokenizer('Hello, my dog is cute', return_tensors='pt').to("cuda")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

4. Clean up by "Ctrl+C" the server process.

## Usage with vLLM

:::tip
To use ServerlessLLM as the load format for vLLM, you need to apply our patch `sllm_store/vllm_patch/sllm_load.patch` to the installed vLLM library. Therefore, please ensure you have applied our `vLLM Patch` as instructed in [installation guide](../getting_started/installation.md).

You may check the patch status by running the following command:
``` bash
./sllm_store/vllm_patch/check_patch.sh
```
If the patch is not applied, you can apply it by running the following command:
```bash
./sllm_store/vllm_patch/patch.sh
```
To remove the applied patch, you can run the following command:
```bash
./sllm_store/vllm_patch/remove_patch.sh
```
:::


Our api aims to be compatible with the `sharded_state` load format in vLLM. Thus, due to the model modifications about the model architecture done by vLLM, the model format for vLLM is **not** the same as we used in transformers. Thus, the `ServerlessLLM format` mentioned in the subsequent sections means the format integrated with vLLM, which is different from the `ServerlessLLM format` used in the previous sections.

Thus, for fist-time users, you have to load the model from other backends and then converted it to the ServerlessLLM format.

1. Download the model from HuggingFace and save it in the ServerlessLLM format:
``` bash
python3 examples/sllm_store/save_vllm_model.py --model_name facebook/opt-1.3b --storage_path $PWD/models --tensor_parallel_size 1

```

You can also transfer the model from the local path compared to download it from network by passing the `--local_model_path` argument.

After downloading the model, you can launch the checkpoint store server and load the model in vLLM through `sllm` load format.

2. Launch the checkpoint store server in a separate process:
```bash
# 'mem_pool_size' is the maximum size of the memory pool in GB. It should be larger than the model size.
sllm-store start --storage-path $PWD/models --mem-pool-size 4GB
```

3. Load the model in vLLM:
```python
from vllm import LLM, SamplingParams

import os

storage_path = os.getenv("STORAGE_PATH", "./models")
model_name = "facebook/opt-1.3b"
model_path = os.path.join(storage_path, model_name)

llm = LLM(
    model=model_path,
    load_format="serverless_llm",
    dtype="float16"
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

## Quantization

ServerlessLLM currently supports bitsandbytes quantization to `nf4`, `fp4`, and `int8` with transformers.

To quantize your model, simply pass in the precision as a parameter in `load_model()` as such:

```
model = load_model("facebook/opt-1.3b", device_map="auto", torch_dtype=torch.float16, storage_path="./models/", fully_parallel=True, quantization="fp4")
```
