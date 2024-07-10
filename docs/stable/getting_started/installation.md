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
# Because this is currelty a private repository, you need to login to github first
# by `gh auth login` and then clone the repository
git clone https://github.com/future-xy/Phantom-component.git
cd Phantom-component
```

```
conda create -n sllm python=3.10 -y
conda activate sllm
pip install -e ".[worker]"
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless_llm_store==0.0.1.dev3
```
