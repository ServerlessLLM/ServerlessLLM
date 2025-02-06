# ServerlessLLM Example Scripts
Please follow the [Installation instructions](https://serverlessllm.github.io/docs/stable/getting_started/installation) to have ServerlessLLM successfully installed.
## Calling Embedding API
This example shows deploying and calling [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) using transformers backend of ServerlessLLM.

### 1. Environment and Service Setup
First and foremost, set the Model Directory `MODEL_FOLDER` where models will be stored:
```bash
export MODEL_FOLDER=/path/to/your/models
```
Secondly, in a new terminal, launch the ServerlessLLM services with docker compose. It's important to note that the model `all-MiniLM-L12-v2` is approximately 0.12GB in size (float32), so you'll need to configure the store server with a memory pool of at least 0.12GB to avoid encountering an Out of Memory error. We recommend setting the memory pool size as large as possible. The memory pool size is set to 4GB by default. **If you would like to change the memory pool size, you need to modify the `command` entry for each `sllm_worker_#` service in `docker-compose.yml` as follows**:

```yaml
command: ["--mem-pool-size", "32GB", "--registration-required", "true"] # This command line option will set a memory pool size of 32GB for each worker node.
```

Afterwards, run docker compose to start the service.

```bash
docker compose up -d
```

### 2. Model Deployment
Now let's deploy the embedding model.

First, create a deployment configuration and save it as a `json` file:
```json
{
    "model": "sentence-transformers/all-MiniLM-L12-v2",
    "backend": "transformers",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10
    },
    "backend_config": {
        "pretrained_model_name_or_path": "",
        "device_map": "auto",
        "torch_dtype": "float32",
        "hf_model_class": "AutoModel"
    }
}
```

We have created the file `transformers_embed_config.json`. Feel free use it. You can also modify it as necessary, or create a new file to suit your requirements.

Next, set the ServerlessLLM Server URL `LLM_SERVER_URL` and deploy this model with the configuration:
```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343/
sllm-cli deploy --config transformers_embed_config.json
```

### 3. Service Request
Post a request by:
```bash
curl http://127.0.0.1:8343/v1/embeddings \
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
                0.03510047867894173,
                0.0419340543448925,
                0.005905449856072664，
                ... # (omit for spacing)
                0.08812830597162247,
                0.03634589537978172,
                0.0021678076591342688,
                0.051571957767009735,
                0.029966454952955246,
                0.02055398002266884
            ]
        }
    ],
    "model": "sentence-transformers/all-MiniLM-L12-v2",
    "usage": {
        "query_tokens": 11,
        "total_tokens": 11
    }
}
```

### 4. Clean Up
In the end, if you would like to delete the model, please run:
```bash
sllm-cli delete sentence-transformers/all-MiniLM-L12-v2
```

To stop the ServerlessLLM services, please run:
```bash
docker compose down
```