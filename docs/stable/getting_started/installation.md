---
sidebar_position: 0
---

# Installations

## Requirements
- OS: Ubuntu 20.04
- Python: 3.10
- GPU: compute capability 7.0 or higher

## Install with pip
TODO

## Install from source
Install the package from source by running the following commands:
```bash
git clone https://github.com/ServerlessLLM/ServerlessLLM
cd ServerlessLLM
```

```
conda create -n sllm python=3.10 -y
conda activate sllm
pip install -e ".[worker]"
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless_llm_store==0.0.1.dev3
```
