---
sidebar_position: 1
---

# Installation with ROCm

## Build the wheel from source and install
ServerlessLLM Store (`sllm-store`) currently supports ROCm. Due to an internal bug in ROCm, serverless-llm-store is only compatible with ROCm version 6.2.0 or higher. Using earlier versions may result in a memory leak, as noted in [issue](https://github.com/ROCm/HIP/issues/3580).

Unfortunately, `pip install .` does not work with ROCm. We suggest you build `sllm-store` wheel and manually install it in your environment.

To build `sllm-store` from source, we suggest you using the docker and build it in ROCm container.

1. Clone the repository and enter the `store` directory:

```bash
git clone git@github.com:ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
```

2. Build the Docker image from `Dockerfile.rocm`

``` bash
docker build -t sllm_store_rocm -f Dockerfile.rocm.
```

3. Build the package inside the ROCm docker container
``` bash
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_builder /bin/bash
export PYTHON_VERSION=310
conda activate py${PYTHON_VERSION} && python setup.py sdist bdist_wheel
```

4. Install pytorch and package in local environment
``` bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
pip install dist/*.whl
```

## Test the installation

Both Python and C++ tests are also available for ROCm platform.

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