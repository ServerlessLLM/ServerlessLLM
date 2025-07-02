---
sidebar_position: 3
---

# Quantization

This example demonstrates the use of quantization within the ServerlessLLM framework to optimize model serving. Quantization is a technique used to reduce the memory footprint and computational requirements of a large language model by representing its weights with lower-precision data types, such as 8-bit integers (int8). This example will showcase how to deploy and serve a quantized model in a ServerlessLLM cluster.

## Pre-requisites

We will use Docker Compose to run a ServerlessLLM cluster in this example. Therefore, please make sure you have read the Quickstart Guide before proceeding.

## Usage
Start a local Docker-based ray cluster using Docker Compose.

## Step 1: Set up the Environment

Create a directory for this example and download the `docker-compose.yml` file.

```bash
mkdir sllm-quantization-example && cd sllm-quantization-example
curl -O https://raw.githubusercontent.com/ServerlessLLM/ServerlessLLM/main/examples/docker/docker-compose.yml

## Step 2: Configuration

Set the Model Directory. Create a directory on your host machine where models will be stored and set the `MODEL_FOLDER` environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

## Step 3: Start the Services

Start the ServerlessLLM services using Docker Compose:

```bash
docker compose up -d
```

This command will start the Ray head node and two worker nodes defined in the `docker-compose.yml` file.

:::tip
Use the following command to monitor the logs of the head node:

```bash
docker logs -f sllm_head
```
:::

## Step 4: Create Quantization and Deployment Configurations

First, we'll generate a standard Hugging Face BitsAndBytesConfig and save it to a JSON file. Then, we'll create a deployment configuration file with these quantization settings embedded in it.

1. Generate the Quantization Config

Create a Python script named `get_config.py` in the current directory with the following content:
```python
# get_config.py
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True) # create the config
quantization_config.to_json_file("quantization_config.json") # save the config into a JSON file

```

Run the script to generate `quantization_config.json`:
```bash
python get_config.py
```


2. Create the Deployment Config

Now, create a file named `quantized_deploy_config.json`. This file tells ServerlessLLM which model to deploy and instructs the backend to use the quantization settings we just created. A template can be found in `sllm/cli/default_config.json`.

```json
{
    "model": "facebook/opt-1.3b",
    "backend": "transformers",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10,
        "keep_alive": 0
    },
    "backend_config": {
        "pretrained_model_name_or_path": "",
        "device_map": "auto",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
        "quantization_config": {
            "_load_in_4bit": true,
            "_load_in_8bit": false,
            "bnb_4bit_compute_dtype": "float32",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": false,
            "llm_int8_enable_fp32_cpu_offload": false,
            "llm_int8_has_fp16_weight": false,
            "llm_int8_skip_modules": null,
            "llm_int8_threshold": 6.0,
            "load_in_4bit": true,
            "load_in_8bit": false,
            "quant_method": "bitsandbytes"
        }
    }
}

```

> Note: Quantization currently only supports the "transformers" backend. Support for other backends will come soon.

## Step 5: Deploy the Quantized Model
With the configuration files in place, deploy the model using the `sllm-cli`.

```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343

sllm-cli deploy --config quantized_deploy_config.json
```

## Step 6: Verify the deployment.
Send an inference to the server to query the model:

```bash
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-1.3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```

To verify the model is being loaded in the desired precision, check the logs (`docker logs sllm_head`). You should see that the model is indeed being loaded in `fp4`.


```log
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 transformers.py:321] load config takes 0.0030286312103271484 seconds
(RoundRobinRouter pid=481) INFO 07-02 20:01:49 roundrobin_router.py:272] []
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 transformers.py:331] load model takes 0.2806234359741211 seconds
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 transformers.py:338] device_map: OrderedDict([('', 0)])
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 transformers.py:345] compute_device_placement takes 0.18753838539123535 seconds
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 transformers.py:376] allocate_cuda_memory takes 0.0020012855529785156 seconds
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 client.py:72] load_into_gpu: transformers/facebook/opt-1.3b, 70b42a05-4faa-4eaf-bb73-512c6453e7fa
(TransformersBackend pid=352, ip=172.18.0.2) INFO 07-02 20:01:49 client.py:113] Model loaded: transformers/facebook/opt-1.3b, 70b42a05-4faa-4eaf-bb73-512c6453e7fa
(TransformersBackend pid=352, ip=172.18.0.2) INFO 07-02 20:01:49 transformers.py:398] restore state_dict takes 0.0007319450378417969 seconds
(TransformersBackend pid=352, ip=172.18.0.2) DEBUG 07-02 20:01:49 transformers.py:411] using precision: fp4
(TransformersBackend pid=352, ip=172.18.0.2) INFO 07-02 20:01:50 client.py:117] confirm_model_loaded: transformers/facebook/opt-1.3b, 70b42a05-4faa-4eaf-bb73-512c6453e7fa
```

You should receive a successful JSON response from the model.

## Step 7: Clean Up

Delete the model deployment by running the following command:

```bash
sllm-cli delete facebook/opt-1.3b
```

If you need to stop and remove the containers, you can use the following commands:

```bash
docker compose down
```


