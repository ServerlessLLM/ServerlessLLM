# Benchmarking for ServerlessLLM Store

Benchmarking suite to measure model loading performance of ServerlessLLM Store against baselines.

## Quick Start

Choose your deployment method:

```bash
# Bare Metal - run directly with sllm-store running
./run-benchmark.sh --model-name facebook/opt-6.7b --benchmark-type random

# Docker - containerized execution
./deploy/docker/run.sh --model-name facebook/opt-6.7b

# Kubernetes - see deploy/k8s/README.md
kubectl apply -f deploy/k8s/configmap.yaml -f deploy/k8s/job.yaml
```

## Benchmark Types

| Type | Description | Use Case |
|------|-------------|----------|
| `random` | Load N different model copies in random order | Simulates multi-model serving scenarios |
| `cached` | Load same model N times | Measures cache/storage performance |

```bash
# Random benchmark (default)
./run-benchmark.sh --benchmark-type random --num-replicas 30

# Cached benchmark
./run-benchmark.sh --benchmark-type cached --num-replicas 5
```

## Common Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `facebook/opt-6.7b` | Model to benchmark |
| `--num-replicas` | `30` | Number of iterations |
| `--mem-pool-size` | `32GB` | sllm-store memory pool |
| `--benchmark-type` | `random` | Test type (random/cached) |
| `--storage-path` | `./models` | Model storage directory |
| `--results-path` | `./results` | Results output directory |
| `--generate-plots` | `false` | Generate visualizations |

## Directory Structure

```
benchmarks/
├── run-benchmark.sh          # Main orchestration script
├── deploy/
│   ├── docker/run.sh         # Docker launcher
│   └── k8s/                  # Kubernetes YAML templates
├── *.py                      # Core Python scripts
├── docs/                     # Detailed documentation
└── results/                  # Benchmark outputs
```

## Deployment Methods

### Bare Metal

Requires `sllm-store` installed. The script handles starting/stopping the server:

```bash
# Ensure models directory exists
mkdir -p ./models

# Run benchmark (starts sllm-store automatically)
./run-benchmark.sh \
    --model-name facebook/opt-6.7b \
    --benchmark-type random \
    --num-replicas 30
```

Results are saved to `./results/` by default.

### Docker

Uses the official `serverlessllm/sllm` image:

```bash
./deploy/docker/run.sh --model-name facebook/opt-6.7b

# With gated model
./deploy/docker/run.sh \
    --model-name meta-llama/Meta-Llama-3-8B \
    --hf-token $HF_TOKEN

# Custom GPU and storage
./deploy/docker/run.sh \
    --gpu-limit 2 \
    --storage-path /mnt/nvme \
    --num-replicas 50
```

See [deploy/docker/README.md](deploy/docker/README.md) for details.

### Kubernetes

Deploy using standard kubectl:

```bash
# Create scripts ConfigMap
kubectl create configmap benchmark-scripts \
    --from-file=run-benchmark.sh \
    --from-file=download_models.py \
    --from-file=test_loading.py \
    --from-file=benchmark_utils.py \
    --from-file=generate_report.py

# Apply config and job
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/job.yaml

# Monitor
kubectl logs -f job/sllm-benchmark
```

See [deploy/k8s/README.md](deploy/k8s/README.md) for details.

## Hardware Requirements

| Test Type | RAM | GPU VRAM | Storage |
|-----------|-----|----------|---------|
| Small models (7B-8B) | 32GB+ | 24GB+ | 500GB+ |
| Large models (66B-70B) | 150GB+ | 160GB+ | 1.5TB+ |

NVMe SSD is highly recommended for optimal performance.

## Output

Results are saved to `--results-path`:

- `summary.txt` - Human-readable summary
- `summary.json` - Machine-readable statistics
- `*_{format}_{replicas}_{type}.json` - Raw benchmark data
- `benchmark.log` - Full execution log

### Example Output

```
============================================================
ServerlessLLM Benchmark Results
============================================================
Model: facebook/opt-6.7b
Replicas: 30
Benchmark Type: random

Loading Time Comparison:
------------------------------------------------------------
Format          Avg (s)      Min (s)      Max (s)      Std Dev
------------------------------------------------------------
SLLM            1.234        1.120        1.450        0.082
SafeTensors     3.456        3.201        3.789        0.156
------------------------------------------------------------
SLLM Speedup: 2.80x faster than SafeTensors
```

## Plotting Results

```bash
python plot.py \
    --models facebook/opt-6.7b meta-llama/Meta-Llama-3-8B \
    --test-name random \
    --num-repeats 30 \
    --results-dir results \
    --output-file images/loading_latency.png
```

## More Information

- [Detailed Benchmark Guide](docs/BENCHMARK.md)
- [Docker Deployment](deploy/docker/README.md)
- [Kubernetes Deployment](deploy/k8s/README.md)

## Contact

Questions? Contact [Y.Fu@ed.ac.uk](mailto:y.fu@ed.ac.uk) or open a GitHub issue.
