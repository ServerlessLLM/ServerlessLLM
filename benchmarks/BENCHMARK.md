# Automated Benchmark for ServerlessLLM

Run end-to-end benchmarks automatically using the official `serverlessllm/sllm` image.

## Quick Start

### Docker (Standalone)

```bash
# Default: facebook/opt-6.7b, 30 replicas, 32GB memory pool, 1 GPU
cd benchmarks
./docker-run.sh

# Custom model
MODEL_NAME=meta-llama/Meta-Llama-3-8B NUM_REPLICAS=50 ./docker-run.sh

# Custom GPU allocation
GPU_LIMIT=2 ./docker-run.sh              # Use 2 GPUs
GPU_LIMIT="all" ./docker-run.sh          # Use all GPUs
GPU_LIMIT='"device=0,1"' ./docker-run.sh # Use specific GPUs

# Custom storage and memory
STORAGE_PATH=/data/nvme MEM_POOL_SIZE=64GB ./docker-run.sh
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

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `facebook/opt-6.7b` | Model to benchmark |
| `NUM_REPLICAS` | `30` | Number of iterations |
| `MEM_POOL_SIZE` | `32GB` | sllm-store memory pool |
| `GPU_LIMIT` | `1` | GPU allocation (1, 2, "all", or "device=X") |
| `BENCHMARK_TYPE` | `random` | Test type (random/single) |
| `STORAGE_PATH` | `/models` | Model storage location |
| `RESULTS_PATH` | `/results` | Results output location |
| `GENERATE_PLOTS` | `false` | Generate plots (requires matplotlib) |
| `KEEP_ALIVE` | `false` | Keep container running after benchmark |

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
- Increase `MEM_POOL_SIZE`
- Use smaller model
- Reduce `NUM_REPLICAS`

**Storage full:**
- Clean old models: `rm -rf $STORAGE_PATH/*`
- Increase storage allocation
- Reduce `NUM_REPLICAS`

**GPU not available:**
```bash
# Verify GPU in pod
kubectl exec <pod> -- nvidia-smi
```

## Advanced Usage

### Multiple Models
```bash
# Run sequentially
for model in facebook/opt-6.7b meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.3; do
  MODEL_NAME=$model ./docker-run.sh
done
```

### Save Results to Different Locations
```bash
# Each run saves to different directory
RESULTS_PATH=./results/opt-6.7b MODEL_NAME=facebook/opt-6.7b ./docker-run.sh
RESULTS_PATH=./results/llama-8b MODEL_NAME=meta-llama/Meta-Llama-3-8B ./docker-run.sh
```

### Debug Mode
```bash
# Keep container alive after benchmark
KEEP_ALIVE=true ./docker-run.sh

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
