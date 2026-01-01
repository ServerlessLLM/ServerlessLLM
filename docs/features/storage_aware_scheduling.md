---
sidebar_position: 0
---

# Storage-Aware Scheduling

In v1-beta, storage-aware scheduling is enabled by default. The system automatically places model instances on nodes that have the model cached, reducing cold-start latency.

## How It Works

Storage-aware placement is built into the Reconciler and StorageManager components:

1. **StorageManager** tracks which models are cached on which nodes
2. **Reconciler** uses `StorageManager.select_best_node()` when creating new instances
3. No special flags or configuration needed - it just works

### Placement Scoring

When the Reconciler needs to create a new instance, `StorageManager.score_node()` evaluates each eligible node:

- **+100 points** if the model is already cached on the node
- **-10 points** per existing instance of the same model on that node (spreading)

The node with the highest score is selected. This ensures:
- Models are placed on nodes where they are cached (fast loading)
- Instances are spread across nodes when possible (load balancing)

### Example Flow

1. User deploys model `meta-llama/Llama-3.1-8B` with `desired_replicas=1`
2. Reconciler sees 0 instances, needs to create 1
3. Reconciler calls `StorageManager.select_best_node()`:
   - Gets list of online workers with enough GPUs
   - Scores each worker (cache hit = +100)
   - Selects highest-scoring node
4. Instance is created on the selected node via Pylet
5. If the model is cached on that node, sllm-store loads it quickly

## Usage

Deploy models normally - storage-aware placement happens automatically:

```bash
# Deploy a model
sllm deploy --model meta-llama/Llama-3.1-8B --backend vllm

# Check status
sllm status
```

Query the model:

```bash
curl http://localhost:8343/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "backend": "vllm",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

## Pre-caching Models

For optimal performance, pre-cache models on worker nodes using `sllm-store`:

```bash
# On a worker node, save a model to local storage
sllm-store save --model meta-llama/Llama-3.1-8B --storage-path /models
```

When sllm-store reports its cached models to the head node, the StorageManager updates its cache view. Future deployments of that model will prefer nodes where it is cached.

## Observability

Check which nodes have which models cached:

```bash
# View sllm-store cache on a worker
sllm-store list --storage-path /models

# Check SLLM database for node storage info (from sllm_head container)
sqlite3 /var/lib/sllm/state.db "SELECT * FROM node_storage;"
```

## Comparison with v1-alpha

| Aspect | v1-alpha | v1-beta |
|--------|----------|---------|
| Enable flag | `--enable-storage-aware` required | Always enabled (default) |
| Component | `StorageAwareScheduler` class | `StorageManager.select_best_node()` |
| Integration | Separate scheduler process | Built into Reconciler |
| Configuration | Config files with placement specs | Automatic, no config needed |
