# Automated Benchmark for ServerlessLLM

Run end-to-end benchmarks automatically using the official `serverlessllm/sllm` image.

## Quick Start

### Docker (Standalone)

```bash
# Default: facebook/opt-6.7b, 30 replicas, 32GB memory pool, 1 GPU
cd benchmarks
./docker-run.sh

# Gated models (e.g., Meta-Llama) - provide HF token
./docker-run.sh --model-name meta-llama/Meta-Llama-3-8B --hf-token $HF_TOKEN

# Or mount existing HF cache (reuses login & downloaded models)
./docker-run.sh --model-name meta-llama/Meta-Llama-3-8B --mount-hf-cache

# Custom model
./docker-run.sh --model-name meta-llama/Meta-Llama-3-8B --num-replicas 50

# Custom GPU allocation
./docker-run.sh --gpu-limit 2                           # Use 2 GPUs
./docker-run.sh --gpu-limit "all"                       # Use all GPUs
./docker-run.sh --gpu-limit '"device=0,1"'              # Use specific GPUs

# Custom storage and memory
./docker-run.sh --storage-path /data/nvme --mem-pool-size 64GB

# All options combined
./docker-run.sh \
    --model-name meta-llama/Meta-Llama-3-8B \
    --num-replicas 50 \
    --mem-pool-size 64GB \
    --gpu-limit 2

# See all options
./docker-run.sh --help
```

### Kubernetes (EIDF)

```bash
# 1. Update benchmark-job.yaml placeholders:
#    - <YOUR_NAMESPACE>
#    - <CPU/MEMORY/GPU resources>
#    - <NVME_MOUNT_PATH>
#    - <RESULTS_PATH>
#    - /path/to/ServerlessLLM/benchmarks

# 2. Deploy configuration
kubectl apply -f k8s/benchmark-configmap.yaml

# 3. Run benchmark
kubectl apply -f k8s/benchmark-job.yaml

# 4. View logs
kubectl logs -f job/sllm-benchmark

# 5. Get results
kubectl cp sllm-benchmark-xxxxx:/results ./results
```

## Configuration

### Command-Line Flags (docker-run.sh)

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--model-name` | `-m` | `facebook/opt-6.7b` | Model to benchmark |
| `--num-replicas` | `-n` | `30` | Number of iterations |
| `--mem-pool-size` | `-p` | `32GB` | sllm-store memory pool |
| `--gpu-limit` | `-g` | `1` | GPU allocation (1, 2, "all", or "device=X") |
| `--benchmark-type` | `-t` | `random` | Test type (random/single) |
| `--storage-path` | `-s` | `/mnt/nvme` | Host storage directory |
| `--results-path` | `-r` | `./results` | Host results directory |
| `--image` | `-i` | `serverlessllm/sllm:latest` | Docker image to use |
| `--hf-token` | - | - | Hugging Face token for gated models |
| `--mount-hf-cache` | - | `false` | Mount ~/.cache/huggingface into container |
| `--generate-plots` | - | `false` | Generate plots (requires matplotlib) |
| `--keep-alive` | - | `false` | Keep container running after benchmark |
| `--help` | `-h` | - | Show help message |

**Note:** Environment variables are still supported for backward compatibility (e.g., `MODEL_NAME=...`, `HF_TOKEN=...`), but CLI flags are recommended and take precedence.

### Custom Models

Edit `k8s/benchmark-configmap.yaml`:
```yaml
data:
  config.env: |
    MODEL_NAME=facebook/opt-6.7b
    NUM_REPLICAS=20
    MEM_POOL_SIZE=16GB
```

Or override in job:
```yaml
env:
- name: MODEL_NAME
  value: "facebook/opt-6.7b"
```

## Output

### Console Output
```
=== ServerlessLLM Benchmark Results ===
Model: facebook/opt-6.7b
Replicas: 30

Loading Time Comparison:
Format          Avg (s)      Min (s)      Max (s)      Std Dev
SLLM            1.234        1.120        1.450        0.082
SafeTensors     3.456        3.201        3.789        0.156
Torch           2.875        2.502        3.201        0.234

SLLM Speedup: 2.80x faster than SafeTensors

Inference Performance:
SLLM Avg Throughput: 72.47 tokens/s
SafeTensors Avg Throughput: 70.78 tokens/s
Torch Avg Throughput: 71.23 tokens/s
```

### Files Generated
- `/results/summary.txt` - Text report
- `/results/summary.json` - JSON statistics
- `/results/benchmark.log` - Full execution log
- `/results/*.json` - Raw benchmark data
- `/results/*.png` - Plots (if enabled)

## What It Does

1. ✅ Auto-detects NVMe storage (if detection script mounted)
2. ✅ Starts sllm-store server in background
3. ✅ Downloads model in three formats (sllm + safetensors + torch)
4. ✅ Runs loading benchmarks with random access pattern
5. ✅ Tests inference performance for all formats
6. ✅ Generates three-way comparison report
7. ✅ Saves all results to mounted volume

## Troubleshooting

**Benchmark fails to start:**
```bash
# Check logs
kubectl logs job/sllm-benchmark

# Check sllm-store log
kubectl exec <pod> -- cat /results/sllm-store.log
```

**Out of memory:**
- Increase memory pool size: `./docker-run.sh --mem-pool-size 64GB`
- Use smaller model
- Reduce replicas: `./docker-run.sh --num-replicas 20`

**Storage full:**
- Clean old models: `rm -rf /mnt/nvme/*` (or your storage path)
- Increase storage allocation
- Reduce replicas: `./docker-run.sh --num-replicas 20`

**GPU not available:**
```bash
# Verify GPU in pod
kubectl exec <pod> -- nvidia-smi
```

**Gated model access denied (401/403 error):**
```bash
# Provide HF token
./docker-run.sh --hf-token $HF_TOKEN --model-name meta-llama/Meta-Llama-3-8B

# Or mount existing HF cache
./docker-run.sh --mount-hf-cache --model-name meta-llama/Meta-Llama-3-8B
```

## Advanced Usage

### Multiple Models
```bash
# Run sequentially with CLI flags
for model in facebook/opt-6.7b meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.3; do
  ./docker-run.sh --model-name "$model"
done
```

### Save Results to Different Locations
```bash
# Each run saves to different directory
./docker-run.sh --model-name facebook/opt-6.7b --results-path ./results/opt-6.7b
./docker-run.sh --model-name meta-llama/Meta-Llama-3-8B --results-path ./results/llama-8b
```

### Debug Mode
```bash
# Keep container alive after benchmark
./docker-run.sh --keep-alive

# Then inspect
docker exec -it <container> bash
```

## Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: 32GB+ RAM (depends on model size and memory pool)
- **Storage**: NVMe SSD recommended
  - opt-1.3b: ~10GB per format (~30GB total for 3 formats)
  - opt-6.7b: ~30GB per format (~90GB total for 3 formats)
  - opt-13b: ~50GB per format (~150GB total for 3 formats)
- **Python**: 3.10+ (included in image)
- **Disk Space**: 3x model size (for three formats: sllm, safetensors, torch)
