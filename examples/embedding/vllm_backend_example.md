# ServerlessLLM Example Scripts
Please follow the [Installation instructions](https://serverlessllm.github.io/docs/stable/getting_started/installation) to have ServerlessLLM successfully installed.
## Calling Embedding API
This example shows deploying and calling [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) using vllm backend of ServerlessLLM.

### 1. Environment and Service Setup
First and foremost, enter the folder where docker compose file is located:
```bash
cd ServerlessLLM/examples/docker/
```
Set the Model Directory `MODEL_FOLDER` where models will be stored:
```bash
export MODEL_FOLDER=/path/to/your/models
```
Secondly, in a new terminal, launch the ServerlessLLM services with docker compose. It's important to note that the model `e5-mistral-7b-instruct` is approximately 14GB in size (float16), so you'll need to configure the store server with a memory pool of at least 14GB to avoid encountering an Out of Memory error. We recommend setting the memory pool size as large as possible. The memory pool size is set to 4GB by default. **If you would like to change the memory pool size, you need to modify the `command` entry for each `sllm_worker_#` service in `docker-compose.yml` as follows**:

```yaml
command: ["-mem_pool_size", "32", "-registration_required", "true"]
```

This command line option will set a memory pool size of 32GB for each worker node.

Afterwards, run docker compose to start the service.

```bash
docker compose up -d --build
```

### 2. Model Deployment
Now let's deploy the embedding model.

First, write your deployment configuration `my_config.json`:
```json
{
    "model": "intfloat/e5-mistral-7b-instruct",
    "backend": "vllm",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10,
        "keep_alive": 0
    },
    "backend_config": {
        "pretrained_model_name_or_path": "intfloat/e5-mistral-7b-instruct",
        "enforce_eager": true,
        "enable_prefix_caching": false,
        "device_map": "auto",
        "torch_dtype": "float16"
    }
}
```
**NOTE:** `enable_prefix_caching: false` and `enforce_eager: true` are necessary for current vLLM version.

Next, set the ServerlessLLM Server URL `LLM_SERVER_URL` and deploy this model with the configuration:
```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343/
sllm-cli deploy --config /path/to/my_config.json
```

### 3. Service Request
Post a request by:
```bash
curl http://127.0.0.1:8343/v1/embeddings \
-H "Content-Type: application/json" \
-d '{
        "model": "intfloat/e5-mistral-7b-instruct",
        "input": [
           "Hi, How are you?"
        ]
    }'
```
You will finally receive the response like:
```bash
{
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [
                0.01345062255859375,
                0.003925323486328125,
                -0.0057220458984375,
                ... # (omit for spacing)
                -0.011810302734375,
                0.032135009765625,
                -0.0028438568115234375,
                -0.0107421875,
                0.01003265380859375
            ]
        }
    ],
    "model": "intfloat/e5-mistral-7b-instruct",
    "usage": {
        "query_tokens": 8,
        "total_tokens": 8
    }
}
```

### 4. Clean Up
In the end, if you would like to delete the model, please run:
```bash
sllm-cli delete intfloat/e5-mistral-7b-instruct
```

To stop the ServerlessLLM services, please run:
```bash
docker compose down
```