# ServerlessLLM Serve Module

This directory contains the core serving infrastructure for ServerlessLLM's HTTP-based distributed architecture.

## Core Components

### Entry Points
- **`commands/serve/sllm_serve.py`** - Main CLI entry point with subcommands `head` and `worker`. No mode flag.

### Head Node Components
- **`api_gateway.py`** - HTTP API server (port 8343) handling client requests (`/v1/chat/completions`, `/v1/embeddings`, `/register`, `/delete`). Routes requests via Redis queues.
- **`dispatcher.py`** - Consumes tasks from Redis model queues and forwards to available worker instances via HTTP.
- **`worker_manager.py`** - Worker lifecycle management. Processes heartbeats, handles registration, and executes scaling operations.
- **`model_manager.py`** - Model lifecycle operations. Manages registration, deletion, and status tracking with "active"/"excommunicado" states.
- **`autoscaler.py`** - Auto-scaling logic that runs every 10 seconds, monitors queue depths, and creates scaling decisions.

### Worker Node Components
- **`worker/api.py`** - Worker HTTP API (port 8001). Exposes endpoints for instance management (`/start_instance`, `/stop_instance`, `/invoke`, `/confirmation`).
- **`worker/heartbeat.py`** - Heartbeat client sending status updates to head node every 15 seconds.
- **`worker/instance_manager.py`** - Local instance lifecycle management. Spawns and manages backend processes.
- **`worker/model_downloader.py`** - Model downloading utilities for transformers and vLLM backends.
- **`worker/hardware_utils.py`** - Hardware detection and resource monitoring utilities.

### Backend Implementations
- **`backends/transformers_backend.py`** - Hugging Face Transformers backend. Spawns subprocess and communicates via HTTP.
- **`backends/vllm_backend.py`** - vLLM backend for high-performance serving. Subprocess-based with HTTP communication.
- **`backends/dummy_backend.py`** - In-memory testing backend for development and debugging.
- **`backends/backend_utils.py`** - Shared utilities, base classes, and process cleanup functions.

### Infrastructure
- **`kv_store.py`** - Redis client with atomic Lua scripts, connection pooling, retry logic, and worker/model state management.
- **`utils.py`** - Shared utilities including HTTP helpers, response formatting, model name generation, and retry mechanisms.
- **`validation.py`** - Request/response validation schemas using Pydantic.
- **`schema.py`** - Data models for workers, models, hardware information, and API schemas.
- **`logger.py`** - Centralized logging configuration.

## Architecture Flow

```
Client Request → API Gateway → Redis Queue → Dispatcher → Worker /invoke → Backend Process
     ↑              ↓                            ↓            ↓              ↓
     └── Response ← Redis Pub/Sub ← Result ← Worker API ← Model Processing
```

### Request Flow
1. **Client** sends request to **API Gateway** (`/v1/chat/completions`)
2. **API Gateway** validates model exists and enqueues task in Redis: `queue:{model_name}:{backend}`
3. **Dispatcher** dequeues task and forwards to worker via `/invoke` endpoint
4. **Worker** routes to appropriate backend instance for processing
5. **Worker** publishes result to Redis channel: `result-channel:{task_id}`
6. **API Gateway** receives result and returns to client

### Worker Registration
1. **Worker** starts with `--node-id` parameter or empty node_id
2. **Worker** sends heartbeat to head `/heartbeat` endpoint every 15 seconds
3. **Worker Manager** processes heartbeat and registers worker in Redis
4. **Worker Manager** sends confirmation to worker `/confirmation` endpoint with assigned node_id
5. **Worker** stores node_id for subsequent heartbeats

### Instance Management
1. **Autoscaler** calculates needed instances: `ceil(queue_length / queue_threshold)` every 10 seconds
2. **Worker Manager** receives scaling decisions from Redis and sends `/start_instance` requests
3. **Instance Manager** spawns backend subprocess with model configuration
4. **Backend** loads model (transformers uses sllm-store, vLLM direct loading)

## Command Usage

### Head Node
```bash
python sllm_serve.py head --host 0.0.0.0 --port 8343 --redis-host localhost --redis-port 6379
```

### Worker Node
```bash
python sllm_serve.py worker --host 0.0.0.0 --port 8001 --node-id UNIQUE_ID --head-node-url http://HEAD_IP:8343
```

## Configuration

### Environment Variables
- `STORAGE_PATH` - Model storage directory (default: "./models")
- Redis connection via head node command-line arguments

### Auto-scaling Configuration (per model)
```json
{
  "auto_scaling_config": {
    "min_instances": 0,
    "max_instances": 1,
    "queue_per_instance_threshold": 5
  }
}
```

## Key Features

- **Atomic Operations**: Redis Lua scripts prevent race conditions in worker registration and model operations
- **Auto-scaling**: Dynamic instance management based on queue depth calculations
- **Health Monitoring**: 15-second heartbeat intervals with 60-second timeout for worker pruning
- **Multiple Backends**: Pluggable backend system (transformers, vLLM, dummy)
- **OpenAI Compatibility**: `/v1/chat/completions` and `/v1/embeddings` endpoints
- **Resource Management**: Proper cleanup of subprocesses and HTTP sessions
- **Error Recovery**: Task requeuing on worker failures and graceful shutdown coordination
- **LoRA Support**: LoRA adapter management for transformers backend