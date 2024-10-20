# sllm-store

## About

`sllm-store` is an internal library of ServerlessLLM that provides high-performance model loading from local storage into GPU memory. You can also install and use this library in your own projects, following our [quick start guide](https://serverlessllm.github.io/docs/stable/store/quickstart).

## Install with pip

``` bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sllm_store==0.0.1.dev5
```

## Install from source
1. Clone the repository and enter the `store` directory

``` bash
git clone git@github.com:ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
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
cd ServerlessLLM/sllm_store
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