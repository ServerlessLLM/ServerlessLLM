# ServerlessLLM Docker Compose Quickstart (v1-beta)

## Architecture

ServerlessLLM v1-beta uses Pylet for GPU instance management:

```
pylet_head (cluster manager)
    │
    ├── pylet_worker (GPU node - runs vLLM/sllm-store instances)
    │
    └── sllm_head (control plane - API Gateway, LoadBalancers, Reconciler)
```

**Key changes from v1-alpha:**
- No Redis required (SQLite for persistence)
- No `sllm start worker` (Pylet manages workers)
- Pylet spawns vLLM instances on-demand

## Quick Start

```bash
# Set model folder
export MODEL_FOLDER=/path/to/models

# Optional: Set HuggingFace token for gated models
export HF_TOKEN=your_token_here

# Build and start
cd examples/docker
docker compose build
docker compose up -d
```

## Verify Services

```bash
# Check pylet head is running
docker logs pylet_head

# Check pylet worker registered
curl http://localhost:8000/workers

# Check sllm head
docker logs sllm_head
curl http://localhost:8343/health
```

Expected health response:
```json
{
  "status": "ok",
  "version": "v1-beta",
  "pylet_connected": true
}
```

## Register a Model

```bash
curl http://localhost:8343/register \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "backend": "vllm",
    "backend_config": {"tensor_parallel_size": 1},
    "auto_scaling_config": {
      "min_instances": 0,
      "max_instances": 1,
      "target_ongoing_requests": 5
    }
  }'
```

## Test Inference

```bash
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m:vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Note:** The first request triggers a cold start. The Reconciler will:
1. Start sllm-store on the worker
2. Start a vLLM instance
3. Wait for the instance to become healthy
4. Forward the request

## Check Status

```bash
curl http://localhost:8343/status | jq .
```

## Configuration

### GPU Configuration

To change the number of GPUs, edit `docker-compose.yml`:

```yaml
pylet_worker:
  environment:
    - GPU_UNITS=4  # Number of GPUs to register
  deploy:
    resources:
      reservations:
        devices:
          - device_ids: ["0", "1", "2", "3"]  # Physical GPU IDs
```

### Environment Variables

| Variable | Service | Description | Default |
|----------|---------|-------------|---------|
| `MODEL_FOLDER` | All | Host path for model storage | Required |
| `HF_TOKEN` | All | HuggingFace token for gated models | Optional |
| `GPU_UNITS` | pylet_worker | Number of GPUs to register | 2 |
| `PYLET_ENDPOINT` | sllm_head | Pylet head URL | http://pylet_head:8000 |
| `SLLM_DATABASE_PATH` | sllm_head | SQLite database path | /var/lib/sllm/state.db |
| `STORAGE_PATH` | All | Model storage path in container | /models |

## Cleanup

```bash
docker compose down
docker volume rm sllm_data  # Remove persistent database
```

## Troubleshooting

### Pylet worker not connecting
```bash
# Check pylet head logs
docker logs pylet_head

# Check worker logs
docker logs pylet_worker
```

### No instances starting
```bash
# Check reconciler logs
docker logs sllm_head | grep -i reconciler

# List workers in pylet
curl http://localhost:8000/workers
```

### Model not found
Ensure models are in the correct format. For vLLM models:
```bash
# Convert model to sllm format
docker exec sllm_head sllm-store save \
  --model facebook/opt-125m \
  --backend vllm \
  --storage-path /models
```

## Model Path Convention

Models must be stored in the format expected by the backend:
- vLLM: `/models/vllm/{model_name}` (e.g., `/models/vllm/facebook/opt-125m`)
- SGLang: `/models/sglang/{model_name}`

Use `sllm-store save` to convert HuggingFace models to the correct format.
