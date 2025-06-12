# ServerlessLLM Example Scripts
Please follow the [Installation instructions](https://serverlessllm.github.io/docs/stable/deployment/single_machine#installation) to have ServerlessLLM successfully installed.
## Calling Embedding API
This example shows deploying and calling [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) using vllm backend of ServerlessLLM.

### 1. Environment and Service Setup
First and foremost, enter the folder where docker compose file is located:
```bash
cd ServerlessLLM/examples/docker/
```
Set the Model Directory `MODEL_FOLDER` where models will be stored:
```bash
export MODEL_FOLDER=/path/to/your/models
```
Secondly, in a new terminal, launch the ServerlessLLM services with docker compose. It's important to note that the model `all-MiniLM-L12-v2` is approximately 0.12GB in size (float32), so you'll need to configure the store server with a memory pool of at least 0.12GB to avoid encountering an Out of Memory error. We recommend setting the memory pool size as large as possible. The memory pool size is set to 4GB by default. **If you would like to change the memory pool size, you need to modify the `command` entry for each `sllm_worker_#` service in `docker-compose.yml` as follows**:

```yaml
command: ["-mem-pool-size", "32", "-registration-required", "true"]
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
    "model": "sentence-transformers/all-MiniLM-L12-v2",
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
        "pretrained_model_name_or_path": "sentence-transformers/all-MiniLM-L12-v2",
        "enforce_eager": true,
        "enable_prefix_caching": false,
        "device_map": "auto",
        "task": "embed",
        "torch_dtype": "float32"
    }
}
```
**NOTE:** `enable_prefix_caching: false` and `enforce_eager: true` are necessary for current vLLM version.

Next, set the ServerlessLLM Server URL `LLM_SERVER_URL` and deploy this model with the configuration:
```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343
sllm deploy --config /path/to/my_config.json
```

### 3. Service Request
Post a request by:
```bash
curl $LLM_SERVER_URL/v1/embeddings \
-H "Content-Type: application/json" \
-d '{
        "model": "sentence-transformers/all-MiniLM-L12-v2",
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
                0.010307748802006245,
                0.060131508857011795,
                0.09968873113393784,
                ... # (omit for spacing)
                -0.04998933523893356,
                -0.016926823183894157,
                0.046083349734544754,
                0.07767919450998306,
                0.029209429398179054,
                -0.08836055546998978
            ]
        }
    ],
    "model": "sentence-transformers/all-MiniLM-L12-v2",
    "usage": {
        "query_tokens": 8,
        "total_tokens": 8
    }
}
```

### 4. Clean Up
In the end, if you would like to delete the model, please run:
```bash
sllm delete sentence-transformers/all-MiniLM-L12-v2
```

To stop the ServerlessLLM services, please run:
```bash
docker compose down
```