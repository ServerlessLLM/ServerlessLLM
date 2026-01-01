# Docker Compose v1-beta End-to-End Test Plan

## Summary
Update `examples/docker/` to support v1-beta architecture with Pylet-based worker management. Test with single worker using 2 GPUs.

---

## Architecture Change

### v1-alpha (Current)
```
redis ─── sllm_head ─── sllm_worker_0 (sllm start worker + sllm-store)
```

### v1-beta (Target)
```
pylet_head ─── pylet_worker (GPU) ─── sllm_head
                    │
                    └── vLLM/sllm-store instances (managed by Pylet)
```

Key differences:
- **No Redis** - SQLite for persistence
- **No `sllm start worker`** - Removed in previous work
- **Pylet** - External cluster manager for GPU instances
- **sllm-store** - Started by Reconciler via Pylet (not entrypoint)

---

## Files to Modify

### 1. `examples/docker/docker-compose.yml`
**Remove:**
- `redis` service (no longer needed)
- `sllm_worker_0` service (replaced by pylet_worker)

**Add:**
- `pylet_head` service - lightweight Python image running Pylet head
- `pylet_worker` service - serverlessllm image with `MODE=PYLET_WORKER`

**Update:**
- `sllm_head` service - v1-beta options, depends on pylet services

### 2. `entrypoint.sh`
**Update HEAD mode:**
- Remove Redis validation
- Add v1-beta CLI options (`--pylet-endpoint`, `--database-path`, `--storage-path`)
- Wait for Pylet head instead of Redis

**Add PYLET_WORKER mode:**
- New mode for GPU worker nodes
- Runs `pylet start --head ... --gpu-units N`
- Uses worker conda env (has vLLM)

### 3. `Dockerfile`
**No changes needed:**
- pylet>=0.3.0 already in requirements.txt
- Both head and worker envs have pylet installed
- Worker env has vLLM for inference

### 4. `examples/docker/README.md`
**Update:**
- Document v1-beta architecture
- New environment variables
- Testing instructions

---

## New docker-compose.yml

**Key insight:** The Pylet worker needs vLLM/sllm-store installed to run inference commands.
Using the existing serverlessllm image (which has vLLM in `worker` env) is the simplest approach.

```yaml
services:
  # Pylet Head Node (cluster manager)
  pylet_head:
    image: python:3.10-slim
    container_name: pylet_head
    command: >
      sh -c "
        pip install pylet httpx &&
        pylet start --host 0.0.0.0
      "
    ports:
      - "8000:8000"
    networks:
      - sllm_network
    healthcheck:
      test: ["CMD", "python3", "-c", "import httpx; httpx.get('http://localhost:8000/workers').raise_for_status()"]
      interval: 5s
      timeout: 5s
      retries: 10
      start_period: 10s

  # Pylet Worker Node (GPU) - uses serverlessllm image for vLLM
  pylet_worker:
    build:
      context: ../../
      dockerfile: Dockerfile
    image: serverlessllm/sllm:latest
    container_name: pylet_worker
    environment:
      - MODE=PYLET_WORKER
      - PYLET_HEAD=pylet_head:8000
      - GPU_UNITS=2
      - STORAGE_PATH=/models
      - HF_TOKEN=${HF_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              device_ids: ["0", "1"]
    ports:
      - "8000-8099:8000-8099"  # vLLM instance ports
    depends_on:
      pylet_head:
        condition: service_healthy
    networks:
      - sllm_network
    volumes:
      - ${MODEL_FOLDER}:/models

  # SLLM Head Node (control plane)
  sllm_head:
    build:
      context: ../../
      dockerfile: Dockerfile
    image: serverlessllm/sllm:latest
    container_name: sllm_head
    environment:
      - MODE=HEAD
      - PYLET_ENDPOINT=http://pylet_head:8000
      - STORAGE_PATH=/models
      - SLLM_DATABASE_PATH=/var/lib/sllm/state.db
      - HF_TOKEN=${HF_TOKEN}
    ports:
      - "8343:8343"
    depends_on:
      pylet_head:
        condition: service_healthy
      pylet_worker:
        condition: service_started
    networks:
      - sllm_network
    volumes:
      - ${MODEL_FOLDER}:/models
      - sllm_data:/var/lib/sllm

networks:
  sllm_network:
    driver: bridge
    name: sllm

volumes:
  sllm_data:
```

---

## Updated entrypoint.sh

```bash
#!/bin/bash
set -e

DEFAULT_HEAD_HOST="0.0.0.0"
DEFAULT_HEAD_PORT="8343"
DEFAULT_STORAGE_PATH="/models"
DEFAULT_DATABASE_PATH="/var/lib/sllm/state.db"
DEFAULT_PYLET_ENDPOINT="http://pylet_head:8000"
DEFAULT_PYLET_HEAD="pylet_head:8000"
DEFAULT_GPU_UNITS="1"

source /opt/conda/etc/profile.d/conda.sh

# HEAD mode: Run SLLM control plane
initialize_head_node() {
  conda activate head

  HEAD_HOST="${HEAD_HOST:-$DEFAULT_HEAD_HOST}"
  HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"
  STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  DATABASE_PATH="${SLLM_DATABASE_PATH:-$DEFAULT_DATABASE_PATH}"
  PYLET_ENDPOINT="${PYLET_ENDPOINT:-$DEFAULT_PYLET_ENDPOINT}"

  # Create database directory
  mkdir -p "$(dirname "$DATABASE_PATH")"

  # Wait for Pylet head to be available
  echo "Waiting for Pylet head at ${PYLET_ENDPOINT}..."
  timeout 60 bash -c "
    until curl -s -o /dev/null -w '%{http_code}' ${PYLET_ENDPOINT}/workers 2>/dev/null | grep -q '200'; do
      sleep 2
    done
  " || echo "WARNING: Could not connect to Pylet head"

  echo "Starting SLLM head..."
  exec sllm start head \
    --host "$HEAD_HOST" \
    --port "$HEAD_PORT" \
    --pylet-endpoint "$PYLET_ENDPOINT" \
    --database-path "$DATABASE_PATH" \
    --storage-path "$STORAGE_PATH" \
    "$@"
}

# PYLET_WORKER mode: Run Pylet worker (for GPU nodes)
initialize_pylet_worker() {
  # Use worker conda env (has vLLM installed)
  conda activate worker

  PYLET_HEAD="${PYLET_HEAD:-$DEFAULT_PYLET_HEAD}"
  GPU_UNITS="${GPU_UNITS:-$DEFAULT_GPU_UNITS}"
  STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"

  # Create storage directory
  mkdir -p "$STORAGE_PATH"

  # Wait for Pylet head to be available
  echo "Waiting for Pylet head at http://${PYLET_HEAD}..."
  timeout 60 bash -c "
    until curl -s -o /dev/null -w '%{http_code}' http://${PYLET_HEAD}/workers 2>/dev/null | grep -q '200'; do
      sleep 2
    done
  " || {
    echo "ERROR: Could not connect to Pylet head at ${PYLET_HEAD}"
    exit 1
  }

  echo "Starting Pylet worker with ${GPU_UNITS} GPUs..."
  exec pylet start --head "$PYLET_HEAD" --gpu-units "$GPU_UNITS"
}

health_check() {
  if [ "$MODE" == "HEAD" ]; then
    HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"
    curl -f "http://localhost:${HEAD_PORT}/health" || exit 1
  else
    # Pylet worker doesn't have HTTP health endpoint, just check process
    pgrep -f "pylet" > /dev/null || exit 1
  fi
}

usage() {
  echo "ServerlessLLM v1-beta Container"
  echo ""
  echo "Modes:"
  echo "  MODE=HEAD          Run SLLM control plane"
  echo "  MODE=PYLET_WORKER  Run Pylet worker (GPU node)"
  echo ""
  echo "HEAD mode env vars:"
  echo "  PYLET_ENDPOINT     Pylet head URL (default: http://pylet_head:8000)"
  echo "  STORAGE_PATH       Model storage path (default: /models)"
  echo "  SLLM_DATABASE_PATH SQLite database path (default: /var/lib/sllm/state.db)"
  echo ""
  echo "PYLET_WORKER mode env vars:"
  echo "  PYLET_HEAD         Pylet head address (default: pylet_head:8000)"
  echo "  GPU_UNITS          Number of GPUs to register (default: 1)"
  echo "  STORAGE_PATH       Model storage path (default: /models)"
}

case "$1" in
  "health")
    health_check
    exit 0
    ;;
  "help")
    usage
    exit 0
    ;;
esac

case "$MODE" in
  "HEAD")
    initialize_head_node "$@"
    ;;
  "PYLET_WORKER")
    initialize_pylet_worker "$@"
    ;;
  *)
    echo "ERROR: MODE must be 'HEAD' or 'PYLET_WORKER'"
    echo ""
    usage
    exit 1
    ;;
esac
```

---

## Testing Procedure

### 1. Build and Start
```bash
cd examples/docker
export MODEL_FOLDER=$(pwd)/models
docker compose build
docker compose up -d
```

### 2. Wait for Services
```bash
# Check pylet head
docker logs pylet_head

# Check pylet worker registered
docker exec pylet_head pylet list-workers

# Check sllm head
docker logs sllm_head
curl http://localhost:8343/health
```

### 3. Register Model
```bash
curl http://localhost:8343/register \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "backend": "vllm",
    "backend_config": {"tensor_parallel_size": 1},
    "auto_scaling_config": {"min_instances": 0, "max_instances": 1}
  }'
```

### 4. Test Inference
```bash
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m:vllm",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### 5. Check Status
```bash
curl http://localhost:8343/status | jq .
```

---

## Implementation Order

1. **Update docker-compose.yml** - New service definitions (pylet_head, pylet_worker, sllm_head)
2. **Update entrypoint.sh** - Add PYLET_WORKER mode, update HEAD mode for v1-beta
3. **Update README.md** - Document v1-beta architecture and usage
4. **Test end-to-end** - Build, start, register model, inference

**Note:** Dockerfile requires no changes - pylet already in requirements.txt

---

## Considerations

### Pylet Worker Image
Using the serverlessllm image for pylet_worker because:
- Already has vLLM installed in `worker` conda env
- Has sllm-store for fast model loading
- Has pylet in requirements.txt
- Consistent with sllm_head image (same build)

### Model Storage
- Shared `/models` volume between pylet_worker and sllm_head
- sllm-store on worker loads models from this path
- Models must be pre-converted to sllm format with `sllm-store save`

### Database Persistence
- SQLite at `/var/lib/sllm/state.db`
- Use Docker volume for persistence across restarts

### Pylet Head
Using lightweight `python:3.10-slim` for pylet_head because:
- Pylet head doesn't need GPU/CUDA
- Just needs Python + pylet package
- Installs pylet at container start (simple, no custom image needed)

### Port Mapping
- Port 8343: SLLM API Gateway (external access)
- Port 8000: Pylet head API (internal only, but exposed for debugging)
- Ports 8000-8099: Reserved for vLLM instances on worker (internal, mapped for health checks)

### Potential Issues
1. **vLLM instances need network access** - Pylet spawns vLLM processes that bind to `$PORT`. Ensure port range is mapped.
2. **Model path consistency** - `command_builder.py` expects `/models/vllm/{model_name}`. Ensure models are stored correctly.
3. **sllm-store lifecycle** - Reconciler starts sllm-store via Pylet. First inference may be slow.
4. **Pylet not in Docker path** - After `pip install pylet`, the `pylet` CLI should be in PATH. Verify this works.

### Dockerfile Changes
- **No changes needed** - pylet>=0.3.0 is already in requirements.txt
- Both head and worker conda envs get pylet installed
- Worker env has vLLM, sllm-store needed for inference

---

## Success Criteria

1. `docker compose up` starts all services
2. Pylet worker registers with pylet head
3. `curl /health` returns 200 with `pylet_connected: true`
4. Model registration creates entry in database
5. Inference request triggers cold start (Reconciler creates instance)
6. Instance becomes ready and serves requests

---

## Implementation Notes (Post-Implementation)

### Fixes Applied

The following bugs were discovered and fixed during testing:

1. **docker-compose.yml**: Removed `--host 0.0.0.0` from `pylet start` (not a valid CLI option)

2. **pylet_client.py**: Fixed pylet 0.3.0 API compatibility:
   - Changed `import pylet.aio` to `import pylet` (no async module)
   - Wrapped sync API calls with `asyncio.to_thread()`
   - Fixed `instances(label=...)` to `instances(labels={...})` (dict, not string)
   - Fixed `worker.worker_id` to `worker.id`
   - Fixed `instance.instance_id` to `instance.id`
   - Fixed worker resource attributes (`gpu` instead of `total_resources.gpu_units`)

### Clean Start Requirement

Always use `docker compose down -v` before `docker compose up` to ensure clean state.
Stale database entries from previous runs can cause issues with orphaned instance records.

### Model Pre-requisite

Models must be saved in sllm format before serving:
```bash
docker exec sllm_head bash -c "
  source /opt/conda/etc/profile.d/conda.sh
  conda activate worker
  sllm-store save --model MODEL_NAME --backend vllm --storage-path /models
"
```
