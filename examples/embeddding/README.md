# ServerlessLLM Example Scripts
Please follow the [Installation instructions](https://serverlessllm.github.io/docs/stable/getting_started/installation) to have ServerlessLLM successfully installed.
## Calling Embedding API
This example shows deploying and calling [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) using ServerlessLLM.

First and foremost, start a local ray cluster with 1 head node and 1 worker node:
```bash
conda activate sllm
ray start --head --port=6379 --num-cpus=4 --num-gpus=0 \
--resources='{"control_node": 1}' --block
```
Start a new terminal and initiate the worker node:
```bash
conda activate sllm-worker
ray start --address=0.0.0.0:6379 --num-cpus=4 --num-gpus=2 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```
Secondly, in a new terminal, launch the ServerlessLLM store server. It's important to note that the model `e5-mistral-7b-instruct` is approximately 14GB in size (float16), so you'll need to configure the store server with a memory pool of at least 14GB to avoid encountering an Out of Memory error. We recommend setting the memory pool size as large as possible.
```bash
conda activate sllm-worker
sllm-store-server --mem_pool_size 14
```
Thirdly, start the ServerlessLLM Serve in another new terminal
```bash
conda activate sllm
sllm-serve start
```
Now let's deploy the embedding model.

First, write your deployment configuration `my_config.json`:
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
        "hf_model_class": "AutoModel"
    }
}
```
Next, deploy this model with the configuration:
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
