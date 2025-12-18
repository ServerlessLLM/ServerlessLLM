# H100 SXM Benchmark Results (December 2025)

## Environment

**Hardware Summary:**
- **Hardware:** 8x NVIDIA H100 SXM
- **Storage:** NVMe SSD with 29.4 GB/s bandwidth
- **Test Date:** December, 2025

**Models Tested:**
- Qwen/Qwen3-32B
- deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
- meta-llama/Llama-3.1-8B-Instruct

## Application Scenarios

The benchmark was conducted in two application scenarios:
- Cached: one model is frequently reused (e.g. local development and testing)
- Random: different model copies are requested in random order (e.g. serverless inference)

## Key Findings

> **SLLM achieves 6-10x speedup over SafeTensors with consistently lower variability across runs.**

## Cached Single Model Load Test

Each model was loaded 5 times consecutively.

| Model | Format | Avg (s) | Min (s) | Max (s) | Std Dev | Speedup |
|-------|--------|---------|---------|---------|---------|---------|
| Qwen/Qwen3-32B | SLLM | 1.253 | 1.165 | 1.508 | 0.144 | **9.95x** |
| Qwen/Qwen3-32B | SafeTensors | 12.466 | 12.382 | 12.591 | 0.097 | - |
| DeepSeek-R1-Distill-Qwen-32B | SLLM | 1.184 | 1.169 | 1.207 | 0.016 | **8.58x** |
| DeepSeek-R1-Distill-Qwen-32B | SafeTensors | 10.165 | 9.973 | 10.324 | 0.134 | - |

![Cached Benchmark](images/cached_benchmark.png)

## Random Multi-Model Load Test

Each test loaded 30 different model copies in random order.

| Model | Format | Avg (s) | Min (s) | Max (s) | Std Dev | Speedup |
|-------|--------|---------|---------|---------|---------|---------|
| Qwen/Qwen3-32B | SLLM | 3.216 | 2.194 | 3.539 | 0.202 | **6.40x** |
| Qwen/Qwen3-32B | SafeTensors | 20.592 | 17.880 | 28.763 | 2.535 | - |
| DeepSeek-R1-Distill-Qwen-32B | SLLM | 3.220 | 2.503 | 3.425 | 0.141 | **5.93x** |
| DeepSeek-R1-Distill-Qwen-32B | SafeTensors | 19.090 | 17.661 | 23.704 | 1.248 | - |
| Llama-3.1-8B-Instruct | SLLM | 0.678 | 0.657 | 0.704 | 0.011 | **6.54x** |
| Llama-3.1-8B-Instruct | SafeTensors | 4.437 | 2.720 | 6.257 | 0.895 | - |

![Random Benchmark](images/random_benchmark.png)

## Raw Data

All raw benchmark data is available in JSON format:
- `cached/` - Contains results from the cached benchmark (5 runs each)
- `random/` - Contains results from the random multi-model benchmark (30 runs each)

Each JSON file follows the naming pattern: `{model_name}_{format}_{num_runs}_{test_type}.json`

