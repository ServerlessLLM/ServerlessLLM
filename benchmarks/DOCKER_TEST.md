# Local Docker Testing

The docker setup uses **exactly the same commands** as the K8s deployment for easy local debugging.

## Quick Start

```bash
cd benchmarks
./docker-run.sh
```

## Custom Configuration

```bash
MODEL_NAME=facebook/opt-1.3b \
NUM_REPLICAS=10 \
MEM_POOL_SIZE=8GB \
STORAGE_PATH=/mnt/nvme \
./docker-run.sh
```

## Key Details

Both docker and K8s versions run:
- Same sllm-store command: `--chunk-size 16MB --num-thread 4 --mem-pool-size 8GB`
- Same benchmark loop with subprocess isolation
- Each format (safetensors, sllm) runs in separate subprocess for GPU memory cleanup

## Debugging GPU Memory Issues

The subprocess approach `( run_format_benchmark "$MODEL_FORMAT" )` ensures:
1. Each format runs in isolated subprocess
2. GPU memory automatically released when subprocess exits
3. Clean state between safetensors and sllm runs

If OOM still occurs on 11th run locally, the issue is in the benchmark Python code itself, not the container/K8s environment.

## Results

Results saved to `./results/`:
- `benchmark.log` - Full log
- `summary.txt` - Summary report
- `*_random.json` - Raw benchmark data
