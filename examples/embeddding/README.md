# ServerlessLLM Example Scripts
Please follow the [Installation instructions](https://serverlessllm.github.io/docs/stable/getting_started/installation) to have ServerlessLLM successfully installed.
## Calling Embedding API
This example shows deploying and calling [gte-Qwen2-1.5B-instruct](https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct) using ServerlessLLM.

This embedding model requires `flash_attn>=2.5.6`. So before you start, you need to do:
```bash
conda activate sllm-worker
pip install flash_attn
```
Now you are ready.

First and foremost, start a local ray cluster with 1 head node and 1 worker node:
```bash
conda activate sllm
ray start --head --port=6379 --num-cpus=4 --num-gpus=0 \
--resources='{"control_node": 1}' --block
```
Start a new terminal and initiate the worker node:
```bash
conda activate sllm-worker
ray start --address=0.0.0.0:6379 --num-cpus=4 --num-gpus=1 \
--resources='{"worker_node": 1, "worker_id_0": 1}' --block
```
Secondly, in a new terminal, launch the ServerlessLLM store server. It's important to note that the model `gte-Qwen2-1.5B-instruct` is approximately 7.2GB in size (float32), so you'll need to configure the store server with a memory pool of at least 7.2GB to avoid encountering an Out of Memory error. We recommend setting the memory pool size as large as possible. So here we set the memory pool size to 20GB.
```bash
conda activate sllm-worker
export CUDA_VISIBLE_DEVICES=0
sllm-store-server --mem_pool_size 20
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
    "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
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
        "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        "task_instruct": "Given a web search query, retrieve relevant passages that answer the query",
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
                -0.01648230291903019,
                0.015597847290337086,
                0.03538760170340538,
                ... # (omit for spacing)
                -0.009428886696696281,
                -0.029391411691904068,
                -0.008450884371995926,
                -0.017801402136683464,
                -0.01637541688978672,
                0.023089321330189705
            ]
        }
    ],
    "model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "usage": {
        "query_tokens": 25,
        "total_tokens": 25
    }
}
```
