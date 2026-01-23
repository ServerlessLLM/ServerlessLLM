# Docker Deployment

Run ServerlessLLM benchmarks in a Docker container using the official `serverlessllm/sllm` image.

## Quick Start

```bash
# From benchmarks/ directory
./deploy/docker/run.sh

# With custom model
./deploy/docker/run.sh --model-name meta-llama/Meta-Llama-3-8B --hf-token $HF_TOKEN

# Cached benchmark
./deploy/docker/run.sh --benchmark-type cached --num-replicas 5
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `facebook/opt-6.7b` | Model to benchmark |
| `--num-replicas` | `30` | Number of iterations |
| `--mem-pool-size` | `32GB` | sllm-store memory pool |
| `--gpu-limit` | `1` | GPU allocation (1, 2, "all", "device=X") |
| `--benchmark-type` | `random` | Test type (random/cached) |
| `--storage-path` | `/mnt/nvme` | Host storage directory |
| `--results-path` | `./results` | Host results directory |
| `--hf-token` | - | Hugging Face token for gated models |
| `--mount-hf-cache` | `false` | Mount ~/.cache/huggingface |
| `--generate-plots` | `false` | Generate plots |
| `--keep-alive` | `false` | Keep container running after completion |

## Requirements

- Docker with NVIDIA GPU support (`nvidia-docker2` or `--gpus` flag support)
- NVIDIA GPU with sufficient VRAM (24GB+ recommended)
- Fast storage (NVMe recommended) mounted at `--storage-path`

## How It Works

1. Mounts the `benchmarks/` directory into the container at `/scripts`
2. Mounts storage and results directories
3. Runs `run-benchmark.sh` inside the container with specified arguments
4. Results are saved to the host `--results-path` directory

## Troubleshooting

**GPU not available:**
```bash
# Verify Docker can see GPUs
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

**Storage full:**
```bash
# Clean storage directory
rm -rf /mnt/nvme/*
```

**Gated model access denied:**
```bash
# Provide HF token
./deploy/docker/run.sh --hf-token $HF_TOKEN --model-name meta-llama/Meta-Llama-3-8B

# Or mount HF cache (if already logged in)
./deploy/docker/run.sh --mount-hf-cache --model-name meta-llama/Meta-Llama-3-8B
```

