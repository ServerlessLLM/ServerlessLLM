# A5000 Benchmark Results (2024)

## Test Environment

For detailed hardware specifications, see [server-specs.md](./server-specs.md).

**Hardware Summary:**
- **CPU:** 2x AMD EPYC 7453
- **GPU:** 8x NVIDIA A5000 24GB
- **Memory:** 1TB DDR4 3200MHz
- **Storage:** 2x 3.84TB NVMe SSD (Intel P5510, PCIe 4.0), RAID 0 configuration

## Benchmark Results

### Small Models - Random Multi-Model Load Test

The benchmark tested several small models (7B-8B parameters) with 30 runs each to measure loading performance in multi-model serving scenarios.

**Models Tested:**
- facebook/opt-6.7b
- meta-llama/Meta-Llama-3-8B
- mistralai/Mistral-7B-v0.3
- google/gemma-7b

![Random Small Loading Latency](random_small_loading_latency.png)

### Large Models

Additional benchmarks were conducted on large models (66B-70B parameter range):
- facebook/opt-66b
- meta-llama/Meta-Llama-3-70B
- mistralai/Mixtral-8x7B-v0.1
- tiiuae/falcon-40b

## Raw Data

This directory contains benchmark results from the A5000 GPU setup. The raw JSON files follow the naming pattern: `{model_name}_{format}_{num_runs}_{test_type}.json`
