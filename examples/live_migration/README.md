# ServerlessLLM Docker Compose Quickstart

```bash
docker compose -f docker-compose.yml -f enable-migration.yml up -d
```

```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343

sllm deploy --config config-qwen-3b.json
sllm deploy --config config-qwen-1.5b.json
```

```bash
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Could you share a story of the history of Computer Science?"}
        ],
        "max_tokens": 1024
    }' &

sleep 3

curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ],
        "max_tokens": 64
    }'
```