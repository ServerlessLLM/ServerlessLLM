# MoE-CAP Evaluation for ServerlessLLM

This example demonstrates how to evaluate MoE (Mixture of Experts) models using the MoE-CAP methodology with ServerlessLLM.

## Overview

The evaluation script (`evaluate_moecap.py`) performs the following steps:

1. **Load Dataset**: Loads benchmark datasets using MoE-CAP data loaders (GSM8K, NuminaMath, LongBench-v2)
2. **Start Recording**: Activates batch statistics recording on ServerlessLLM
3. **Run Inference**: Sends inference requests to the deployed model via HTTP
4. **Stop Recording**: Deactivates recording
5. **Collect Statistics**: Retrieves batch execution statistics
6. **Calculate Metrics**: Computes MoE-CAP performance metrics

**Key Feature**: This script uses MoE-CAP's battle-tested data loaders and configs, but sends requests directly to ServerlessLLM's HTTP API instead of using MoE-CAP's profiler. This provides maximum compatibility with ServerlessLLM while leveraging MoE-CAP's dataset utilities.

## Prerequisites

1. **Install MoE-CAP** (for data loaders):
   ```bash
   cd /home/jysc/MoE-CAP-Real
   pip install -e .
   ```

2. **Deploy Model with vllm_moecap Backend**:
   ```bash
   # Make sure your model is deployed with backend: "vllm_moecap"
   sllm deploy --model Qwen/Qwen3-30B-A3B --config moecap_config.json
   ```

## Usage

### Basic Evaluation

```bash
python evaluate_moecap.py \
    --model Qwen/Qwen3-30B-A3B \
    --dataset gsm8k \
    --server-url http://127.0.0.1:8500 \
    --output-dir ./results
```

### With Custom Parameters

```bash
python evaluate_moecap.py \
    --model Qwen/Qwen3-30B-A3B \
    --dataset numinamath \
    --num-samples 100 \
    --num-gpus 4 \
    --max-tokens 1024 \
    --temperature 0.7 \
    --clear-before \
    --output-dir ./results/numina_eval
```

### Arguments

- `--model`: Model name (must match deployed model)
- `--dataset`: Dataset name (gsm8k, numinamath, longbench_v2)
- `--num-samples`: Number of samples to evaluate (default: all)
- `--num-gpus`: Number of GPUs used for inference (default: 1, needed for advanced metrics)
- `--server-url`: ServerlessLLM server URL (default: http://127.0.0.1:8500)
- `--output-dir`: Output directory for results (default: ./moecap_results)
- `--max-tokens`: Maximum tokens per generation (default: 512)
- `--temperature`: Sampling temperature (default: 0.0)
- `--clear-before`: Clear previous recordings before starting

## Supported Datasets

The script uses MoE-CAP's data loaders, which support:

- **gsm8k**: Grade school math problems (8.5K test samples)
- **numinamath**: Mathematical reasoning dataset
- **longbench_v2**: Long-context understanding benchmarks

All datasets are automatically downloaded via HuggingFace `datasets` library.

## Output Files

The script generates four files in the output directory:

1. **metrics.json**: Combined performance and accuracy metrics
   - **Batch metrics**:
     - Basic: Batch statistics (throughput, latency, batch sizes)
     - Advanced (MoE-CAP): Memory bandwidth utilization (SMBU), FLOPS utilization (SMFU), KV cache size, TTFT, TPOT
   - **Accuracy metrics**: Exact match (EM) score and correctness counts

2. **batch_data.json**: Raw batch execution statistics
   - Per-batch timing information
   - Batch sizes
   - Sequence lengths
   - Expert activation patterns (if available)

3. **inference_outputs.json**: Model inference results
   - Input prompts
   - Expected answers
   - Model responses

4. **config.json**: Evaluation configuration
   - Model name
   - Dataset name
   - Number of samples
   - Generation parameters

## Example Metrics Output

```json
{
  "batch_metrics": {
    "total_batches": 50,
    "total_requests": 100,
    "avg_batch_size": 2.0,
    "max_batch_size": 4,
    "min_batch_size": 1,
    "avg_latency_ms": 125.5,
    "max_latency_ms": 450.2,
    "min_latency_ms": 85.3,
    "throughput_rps": 15.8,
    "total_time_s": 6.275,
    "prefill_smbu": 0.65,
    "prefill_smfu": 0.42,
    "decoding_smbu": 0.58,
    "decoding_smfu": 0.35,
    "kv_size": 2048.5,
    "decoding_throughput": 18.5,
    "prefill_tp": 450.2,
    "ttft": 0.125,
    "tpot": 0.054
  },
  "accuracy_metrics": {
    "exact_match": 0.85,
    "correct": 85,
    "total": 100
  }
}
```

**Metric Descriptions:**
- `prefill_smbu/decoding_smbu`: Memory bandwidth utilization during prefill/decode phases
- `prefill_smfu/decoding_smfu`: FLOPS utilization during prefill/decode phases
- `kv_size`: Average KV cache size in MB
- `ttft`: Time to first token (seconds)
- `tpot`: Time per output token (seconds)
- `prefill_tp`: Prefill throughput (tokens/sec)
- `decoding_throughput`: Decode throughput (requests/sec)

## API Endpoints

The script uses the following ServerlessLLM API endpoints:

- `POST /start_batch_recording` - Start recording batch statistics
- `POST /stop_batch_recording` - Stop recording
- `POST /dump_batch_recording` - Retrieve recorded statistics
- `GET /batch_recording_status` - Check recording status
- `POST /clear_batch_recording` - Clear previous recordings
- `POST /v1/chat/completions` - Send inference requests

## Troubleshooting

### "Model does not support batch recording"

Make sure your model is deployed with `backend: "vllm_moecap"` in the config file:

```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "backend": "vllm_moecap",
  ...
}
```

### "No batch data collected"

This can happen if:
1. The model wasn't deployed with `vllm_moecap` backend
2. No inference requests were processed during recording
3. Recording wasn't started successfully

### Rate Limiting

If you're sending many requests, you may want to add delays:
- Modify the `time.sleep(0.1)` value in the script
- Or implement batch request sending

## Configuration File

The `moecap_config.json` file should specify the `vllm_moecap` backend:

```json
{
  "model": "Qwen/Qwen3-30B-A3B",
  "backend": "vllm_moecap",
  "num_gpus": 4,
  "backend_config": {
    "tensor_parallel_size": 4,
    "max_model_len": 40960,
    "enforce_eager": false,
    "enable_prefix_caching": true,
    "trust_remote_code": true,
    "gpu_memory_utilization": 0.9
  },
  "auto_scaling_config": {
    "metric": "concurrency",
    "target": 1,
    "min_instances": 0,
    "max_instances": 1
  }
}
```

## Notes

- The script uses **MoE-CAP's data loaders** for dataset handling (more robust and tested)
- It uses **MoE-CAP's accuracy metrics** utilities for answer extraction and exact match computation
- It sends requests via **ServerlessLLM's HTTP API** (no need for MoE-CAP profiler)
- This hybrid approach provides the best of both worlds: reliable data loading + accurate metric computation + direct ServerlessLLM integration
- The script includes a small delay between requests (100ms) to avoid overwhelming the server
- Answer extraction is dataset-specific (GSM8K uses `####` pattern, etc.) via MoE-CAP utilities
- Model info can be retrieved using `HFModelInfoRetriever` from MoE-CAP if needed for analysis
