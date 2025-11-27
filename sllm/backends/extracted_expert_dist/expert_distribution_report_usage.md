# vLLM Expert Distribution Report - Quick Start Guide

## 1. Start the Server

Simply add the `--enable-expert-distribution-metrics` flag when starting the server. This automatically enables recording in `per_pass` mode and writes logs to a file.

```bash
python -m moe_cap.systems.vllm Qwen/Qwen1.5-MoE-A2.7B \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --port 8000 \
  --enable-expert-distribution-metrics
```

## 2. Check the Logs

Send some inference requests to the server, and the expert distribution metrics will be recorded.
```bash
curl http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
  "model": "Qwen/Qwen1.5-MoE-A2.7B",
  "prompt": "Write a Python program of a tik tak toe.",
  "max_tokens": 3000,
  "temperature": 0
}'
```

The server automatically writes expert distribution metrics to a JSONL file in the `logs/expert_distribution` directory. The filename includes the model name and a timestamp.


**Example Content:**
```json
{"forward_pass_id": 1, "batch_size": 16, "latency": 0.023, "seq_lens_sum": 128, "forward_mode": "prefill", "expert_activation": 26.4, "expert_utilization": 0.44}
{"forward_pass_id": 2, "batch_size": 1,  "latency": 0.006, "seq_lens_sum": 129, "forward_mode": "decode",  "expert_activation": 4.0,  "expert_utilization": 0.06}
```

## 3. API Controls (Optional)

You can also control the recording and get summaries via API.

### Dump Summary
Get a quick summary of expert activation stats (average activation for prefill/decode) without reading the full log file.

```bash
curl -X POST "http://localhost:8000/dump_expert_distribution"
```

**Example Response:**
```json
{
  "status": "success",
  "num_workers": 4,
  "summary": {
    "average_expert_activation_prefill": 26.41,
    "average_expert_activation_decode": 4.0,
    "workers": [
      {
        "rank": 0,
        "num_experts": 60,
        "sample_record": {
           "expert_activation": 4.0,
           "expert_utilization": 0.0667
        }
      }
    ],
    "jsonl_file": "/abs/path/to/logs/expert_distribution/.../expert_distribution_record.jsonl"
  }
}
```

### Control Recording
If you want to pause or restart recording for a specific test case:

```bash
# Stop/Pause recording
curl -X POST "http://localhost:8000/stop_expert_distribution"

# Start/Restart recording (clears previous in-memory data, but appends to file)
curl -X POST "http://localhost:8000/start_expert_distribution"
```

