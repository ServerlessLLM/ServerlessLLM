# Detailed Benchmark Guide

This document provides detailed information about the ServerlessLLM benchmark system.

## Architecture

The benchmark system consists of:

1. **Core Python Scripts** - Measurement and reporting logic
2. **Orchestration Script** - `run-benchmark.sh` handles the full pipeline
3. **Deployment Launchers** - Docker and K8s wrappers

```
run-benchmark.sh
    ├── download_models.py (download & save models)
    ├── test_loading.py (measure loading times)
    ├── benchmark_utils.py (utilities)
    └── generate_report.py (create reports)
```

## What the Benchmark Measures

### Loading Time

Time to load a model from storage to GPU memory, comparing:
- **SLLM format**: ServerlessLLM's optimized format
- **SafeTensors format**: HuggingFace's standard format

### Inference Performance

After loading, the benchmark runs a short inference to verify the model works and measure throughput.

## Benchmark Types

### Random Load Test

Simulates multi-model serving where different models are requested randomly.

**How it works:**
1. Downloads N copies of the model (one per replica)
2. Loads models in random order
3. Measures each load time

**Use case:** Testing model swapping in serverless inference.

```bash
./run-benchmark.sh --benchmark-type random --num-replicas 30
```

### Cached Load Test

Measures repeated loading of the same model.

**How it works:**
1. Downloads 1 copy of the model
2. Performs warmup load (not measured)
3. Loads the same model N times
4. Measures each load time

**Use case:** Testing storage/cache performance, requires less disk space.

```bash
./run-benchmark.sh --benchmark-type cached --num-replicas 5
```

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `facebook/opt-6.7b` | HuggingFace model ID |
| `--num-replicas` | `30` | Number of load iterations |
| `--mem-pool-size` | `32GB` | sllm-store memory pool |
| `--benchmark-type` | `random` | Test type (random/cached) |
| `--storage-path` | `/models` | Model storage directory |
| `--results-path` | `/results` | Results output directory |
| `--generate-plots` | `false` | Generate visualizations |
| `--keep-alive` | `false` | Keep running after completion |

## Output Files

Results are saved to `--results-path`:

| File | Description |
|------|-------------|
| `summary.txt` | Human-readable report |
| `summary.json` | Machine-readable statistics |
| `{model}_{format}_{replicas}_{type}.json` | Raw per-load measurements |
| `benchmark.log` | Full execution log |
| `sllm-store.log` | Server logs |

## Example Results

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

Inference Performance:
------------------------------------------------------------
SLLM Avg Throughput: 72.47 tokens/s
SafeTensors Avg Throughput: 70.78 tokens/s
============================================================
```

## Running Multiple Benchmarks

### Multiple Models

```bash
for model in facebook/opt-6.7b meta-llama/Meta-Llama-3-8B; do
    ./deploy/docker/run.sh \
        --model-name "$model" \
        --results-path "./results/${model//\//_}"
done
```

### Both Benchmark Types

```bash
for type in random cached; do
    ./deploy/docker/run.sh \
        --model-name facebook/opt-6.7b \
        --benchmark-type "$type" \
        --results-path "./results/$type"
done
```

## Plotting Results

Generate comparison plots from benchmark results:

```bash
python plot.py \
    --models facebook/opt-6.7b meta-llama/Meta-Llama-3-8B \
    --test-name random \
    --num-repeats 30 \
    --results-dir results \
    --output-file images/loading_latency.png
```

## Troubleshooting

### Out of Memory

- Reduce `--mem-pool-size`
- Reduce `--num-replicas`
- Use smaller model

### Storage Full

- Clean storage directory before running
- Reduce `--num-replicas`
- Use `cached` benchmark type (requires less space)

### GPU Memory Not Released

The benchmark runs each format in a subprocess to ensure GPU memory cleanup. If issues persist, check `benchmark.log` for errors.

### Gated Model Access

For models like Meta-Llama that require acceptance:
1. Accept terms on HuggingFace website
2. Provide token: `--hf-token $HF_TOKEN`
3. Or mount HF cache: `--mount-hf-cache` (Docker only)

## Hardware Recommendations

| Model Size | RAM | GPU VRAM | Storage |
|------------|-----|----------|---------|
| 7B-8B | 32GB | 24GB | 100GB |
| 13B | 64GB | 48GB | 200GB |
| 66B-70B | 150GB | 160GB | 1TB |

NVMe SSD strongly recommended for accurate performance measurements.
