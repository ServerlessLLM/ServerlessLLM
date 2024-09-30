# ServerlessLLM Example Scripts
Please follow the [quickstart instructions](https://serverlessllm.github.io/docs/stable/getting_started/quickstart) to start a Ray cluster, a Serverless LLM store and a Serverless LLM Serve.
## Calling Embedding API
This example shows deploying and calling [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) using Serverless LLM. \
First and foremost, write your deployment configuration `my_config.json`:
```json
{
    "model": "intfloat/e5-mistral-7b-instruct",
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
        "torch_dtype": "float16",
        "hf_model_type": "auto-model"
    }
}
```
Secondly, deploy this model with the transformers backend:
```bash
conda activate sllm
sllm-cli deploy --config /path/to/my_config.json
```
Then post a request.
```bash
curl http://127.0.0.1:8343/v1/embeddings \
-H "Content-Type: application/json" \
-d '{
        "model": "intfloat/e5-mistral-7b-instruct",
        "task_instruct": "Given a question, retrieve passages that answer the question",
        "query": [
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
                0.0027561187744140625,
                0.00395965576171875,
                -0.004638671875,
                ... # (omit for spacing)
                0.0007066726684570312,
                -0.0082550048828125,
                0.0109710693359375,
                0.00965118408203125,
                -0.0013055801391601562,
                0.005157470703125
            ]
        }
    ],
    "model": "intfloat/e5-mistral-7b-instruct",
    "usage": {
        "query_tokens": 23,
        "total_tokens": 23
    }
}
```
