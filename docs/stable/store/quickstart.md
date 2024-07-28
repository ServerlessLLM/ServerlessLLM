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
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless_llm_store==0.0.1.dev3
```

### Install from source
1. Clone the repository and enter the `store` directory

``` bash
git clone git@github.com:ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/serverless_llm/store
```

2. Install the package from source

```bash
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
from serverless_llm_store import save_model

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
sllm-store-server --storage_path $PWD/models --mem_pool_size 32
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
from serverless_llm_store import load_model

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