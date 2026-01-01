# Docker Compose Deployment (v1-beta)

This guide covers deploying ServerlessLLM using Docker Compose with the v1-beta Pylet-based architecture.

## Architecture

```
pylet_head (cluster manager)
    |
    +-- pylet_worker (GPU node - runs vLLM instances)
    |
    +-- sllm_head (control plane - API Gateway, Reconciler, Autoscaler)
```

**Key differences from v1-alpha:**
- No Redis required (SQLite for persistence)
- Pylet manages GPU workers and spawns vLLM instances on-demand
- Models must be pre-saved in sllm format

## Quick Start

```bash
cd examples/docker

# Set model storage path
export MODEL_FOLDER=$(pwd)/models
mkdir -p $MODEL_FOLDER

# Optional: HuggingFace token for gated models
export HF_TOKEN=your_token_here

# Build and start (clean)
docker compose up -d
```

## Clean Start vs Persistent Data

### Clean Start (Default Recommendation)

For development and testing, start fresh each time:

```bash
# Full cleanup - removes all containers, volumes, and networks
docker compose down -v

# Start fresh
docker compose up -d
```

The `-v` flag removes named volumes, ensuring:
- Database is reset (no stale model registrations)
- No orphaned instance records from previous runs
- Clean Pylet worker state

### Persistent Data

For production deployments where you want to preserve state across restarts:

```bash
# Stop without removing volumes
docker compose down

# Restart - preserves database and model cache
docker compose up -d
```

**Warning:** If workers change between restarts, you may see warnings about orphaned instances. The system will clean these up automatically, but a clean start is recommended when changing configurations.

## Pre-requisites: Model Preparation

Models must be saved in sllm format before they can be served. This is a **one-time operation** per model.

```bash
# Save model to sllm format (run inside container or on host with sllm-store)
docker exec sllm_head bash -c "
  source /opt/venvs/sllm-store/bin/activate
  sllm-store save \
    --model facebook/opt-125m \
    --backend vllm \
    --storage-path /models
"
```

This creates the model files at `/models/vllm/facebook/opt-125m`.

## Usage

### 1. Register a Model

```bash
curl http://localhost:8343/deployments \
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

### 2. Test Inference

```bash
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "backend": "vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Note:** The first request triggers a cold start (~30-60s for small models).

### 3. Check Status

```bash
# Health check
curl http://localhost:8343/health

# Full status (models, instances, workers)
curl http://localhost:8343/status | jq .
```

## Configuration

### Environment Variables

| Variable | Service | Description | Default |
|----------|---------|-------------|---------|
| `MODEL_FOLDER` | All | Host path for model storage | **Required** |
| `HF_TOKEN` | All | HuggingFace token for gated models | Optional |
| `GPU_UNITS` | pylet_worker | Number of GPUs to register | 2 |
| `PYLET_ENDPOINT` | sllm_head | Pylet head URL | http://pylet_head:8000 |
| `SLLM_DATABASE_PATH` | sllm_head | SQLite database path | /var/lib/sllm/state.db |

### GPU Configuration

Edit `docker-compose.yml` to change GPU allocation:

```yaml
pylet_worker:
  environment:
    - GPU_UNITS=4  # Number of GPUs to register with Pylet
  deploy:
    resources:
      reservations:
        devices:
          - device_ids: ["0", "1", "2", "3"]  # Physical GPU IDs
```

## Troubleshooting

### Services won't start

```bash
# Check logs
docker logs pylet_head
docker logs pylet_worker
docker logs sllm_head
```

### Pylet worker not connecting

```bash
# Check pylet head is healthy
curl http://localhost:8000/workers

# Check worker registration
docker logs pylet_worker | grep -i register
```

### Cold start timeout

Common causes:
1. **Model not saved** - Run `sllm-store save` first
2. **GPU memory** - Reduce model size or increase GPU allocation
3. **Network issues** - Check container networking

```bash
# Check what's happening during cold start
docker logs sllm_head | grep -i reconciler
docker logs pylet_worker | tail -20
```

### Stale data from previous runs

```bash
# Full cleanup
docker compose down -v
docker volume rm docker_sllm_data 2>/dev/null

# Start fresh
docker compose up -d
```

## Complete Cleanup

Remove all ServerlessLLM Docker resources:

```bash
# Stop and remove everything
docker compose down -v

# Remove any orphaned resources
docker rm -f pylet_head pylet_worker sllm_head 2>/dev/null
docker volume rm docker_sllm_data 2>/dev/null
docker network rm sllm 2>/dev/null
```
