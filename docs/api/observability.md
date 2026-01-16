---
sidebar_position: 3
---

# Observability

ServerlessLLM exposes Prometheus metrics from the head node and can emit a
file-based service discovery (file_sd) list of healthy backend endpoints.

## Enable Prometheus

Metrics are always available at `/metrics` on the head node. To enable
file-based service discovery, set a file_sd output path:

```bash
sllm start --prometheus-sd-path /var/lib/sllm/prometheus_sd.json
```

Environment variables:

- `SLLM_PROMETHEUS_SD_PATH=/var/lib/sllm/prometheus_sd.json`

## Metrics

The head node serves Prometheus metrics at `/metrics`. Initial metrics
include:

- `sllm_router_requests_total{deployment_id,backend}`
- `sllm_router_instances{deployment_id,backend}`

## Prometheus file_sd

ServerlessLLM writes a file_sd JSON list of healthy inference endpoints so
Prometheus can discover backends (vLLM/SGLang). Example:

```json
[
  {
    "labels": {
      "backend": "vllm",
      "deployment_id": "facebook/opt-1.3b:vllm"
    },
    "targets": [
      "10.0.0.12:8000",
      "10.0.0.13:8000"
    ]
  }
]
```

Prometheus scrape config snippet:

```yaml
scrape_configs:
  - job_name: sllm-head
    static_configs:
      - targets: ["sllm-head:8343"]
    metrics_path: /metrics

  - job_name: sllm-backends
    file_sd_configs:
      - files:
          - /var/lib/sllm/prometheus_sd.json
```
