# serverless_llm-store

## About

`sllm-store` is an internal library of ServerlessLLM that provides high-performance model loading from local storage into GPU memory. You can also install and use this library in your own projects, following our [quick start guide](https://serverlessllm.github.io/docs/stable/store/quickstart).

## Install with pip

``` bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless_llm_store==0.0.1.dev3
```

## Install from source
1. Clone the repository and enter the `store` directory

``` bash
git clone git@github.com:ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/serverless_llm/store
```

2. Install the package from source

```bash
pip install .
```

## Build the wheel from source

To build `sllm-store` from source, we suggest you using the docker and build it in NVIDIA container.

1. Clone the repository and enter the `store` directory:

```bash
git clone git@github.com:ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/serverless_llm/store
```

2. Build the Docker image from `Dockerfile.builder`

``` bash
docker build -t sllm_store_builder -f Dockerfile.builder .
```

3. Build the package inside the NVIDIA docker container
``` bash
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_builder /bin/bash
export PYTHON_VERSION=310
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
conda activate py${PYTHON_VERSION} && python setup.py sdist bdist_wheel
``` 

4. Install the package in local environment
``` bash
pip install dist/*.whl
```

5. Following the [quick start guide](https://serverlessllm.github.io/docs/stable/store/quickstart) to use the library.

## ROCm Support

Currently, `sllm-store` only support build ROCm wheel from source. We now only tested on PyTorch 2.3.0 with ROCm 6.2.0. Versions lower than ROCm 6.2.0 will face shared memory not released issue due to the memory leak in `hipIpcCloseMemHandle`. For more details, please refer to this [ROCm issue](https://github.com/ROCm/HIP/issues/3580).

To build the ROCm version of `sllm-store`, we recommend you to use the docker and build it in ROCm container.

1. Build the Docker image from `Dockerfile.rocm`

``` bash
docker build -t sllm_store_rocm -f Dockerfile.rocm .
```

2. Build the package inside the ROCm docker container
``` bash
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_rocm /bin/bash
export PYTHON_VERSION=310
conda activate py${PYTHON_VERSION} && python setup.py sdist bdist_wheel
```

3. After that, you can install the ROCm version of `sllm-store` in your local environment.

``` bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/rocm6.0
pip install dist/*.whl
```