---
sidebar_position: 1
---

# Installation with ROCm (Experimental)

## Build the wheel from source and install
ServerlessLLM Store (`sllm-store`) currently provides experimental support for ROCm platform. Due to an internal bug in ROCm, serverless-llm-store may face a GPU memory leak in ROCm before version 6.2.0, as noted in [issue](https://github.com/ROCm/HIP/issues/3580).

Currently, `pip install .` does not work with ROCm. We suggest you build `sllm-store` wheel and manually install it in your environment.

To build `sllm-store` from source, we suggest you using the docker and build it in ROCm container.

1. Clone the repository and enter the `store` directory:

```bash
git clone git@github.com:ServerlessLLM/ServerlessLLM.git
cd ServerlessLLM/sllm_store
```

2. Build the Docker image from `Dockerfile.rocm`. The `Dockerfile.rocm` is build on top of `rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0` image.

``` bash
docker build -t sllm_store_rocm -f Dockerfile.rocm .
```

3. Build the package inside the ROCm docker container
``` bash
docker run -it --rm -v $(pwd)/dist:/app/dist sllm_store_rocm /bin/bash
python setup.py sdist bdist_wheel
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

### End to end tests

1. Save the `faceboo/opt-1.3b` model in `./models` directory

``` bash
python3 examples/sllm_store/load_transformers_model.py --model_name facebook/opt-1.3b --storage_path ./models
```

2. Start the `sllm-store` server

```bash
sllm-store-server
```

3. Load the model and run the inference

```bash
python3 examples/sllm_store/save_transformers_model.py --model_name facebook/opt-1.3b --storage_path ./models

```

Expected Output:
``` bash
```

## Known issues

1. GPU memory leak in ROCm before version 6.2.0.

This issue is due to an internal bug in ROCm. After the inference instance is completed, the GPU memory is still occupied and not related. For more information, please refer to [issue](https://github.com/ROCm/HIP/issues/3580).

2. vLLM v0.5.0.post1 can not be built with ROCm 6.2.0
