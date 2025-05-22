---
sidebar_position: 0
---

# Installation

## Requirements
- OS: Ubuntu 20.04
- Python: 3.10
- GPU: compute capability 7.0 or higher

## Installing with pip
```bash
# On the head node
conda create -n sllm python=3.10 -y
conda activate sllm
pip install serverless-llm
pip install serverless-llm-store

# On a worker node
conda create -n sllm-worker python=3.10 -y
conda activate sllm-worker
pip install serverless-llm[worker]
pip install serverless-llm-store
```

:::note
If you plan to use vLLM with ServerlessLLM, you need to apply our patch to the vLLM repository. Refer to the [vLLM Patch](#vllm-patch) section for more details.
:::


## Installing from source
To install the package from source, follow these steps:
```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM
```

```
# On the head node
conda create -n sllm python=3.10 -y
conda activate sllm
pip install -e .
cd sllm_store && rm -rf build
# Installing `sllm_store` from source can be slow. We recommend using pip install.
pip install .

# On a worker node
conda create -n sllm-worker python=3.10 -y
conda activate sllm-worker
pip install -e ".[worker]"
cd sllm_store && rm -rf build
# Installing `sllm_store` from source can be slow. We recommend using pip install.
pip install .
```

# vLLM Patch
To use vLLM with ServerlessLLM, you need to apply our patch located at `sllm_store/vllm_patch/sllm_load.patch` to the vLLM repository. to the vLLM repository.
The patch has been tested with vLLM version `0.6.6`.

You can apply the patch by running the following script:
```bash
conda activate sllm-worker
./sllm_store/vllm_patch/patch.sh
```
