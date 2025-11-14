# Kubernetes Deployment for ServerlessLLM Benchmarks

Automated benchmark system for EIDF GPU cluster.

## Quick Start

```bash
# Deploy benchmark
NS=eidf230ns k8s/deploy-benchmark.sh

# Monitor progress
NS=eidf230ns k8s/monitor-benchmark.sh

# Check queue status
NS=eidf230ns k8s/monitor-queue.sh
```

See [QUICK_DEPLOY.md](QUICK_DEPLOY.md) for full guide.

## Files

### Deployment
- `benchmark-job-eidf.yaml` - Kubernetes Job manifest for EIDF
- `benchmark-configmap.yaml` - Benchmark configuration
- `benchmark-scripts-configmap.yaml` - Benchmark scripts (downloads from GitHub)
- `deploy-benchmark.sh` - Auto-deployment script

### Monitoring
- `monitor-benchmark.sh` - Monitor benchmark progress
- `monitor-queue.sh` - Check Kueue queue status

### Documentation
- `QUICK_DEPLOY.md` - Quick deployment guide
- `EIDF_SETUP.md` - EIDF-specific setup details
- `QUEUE_MONITORING.md` - Queue monitoring guide

## Architecture

Uses official `serverlessllm/sllm:latest` image with:
- Scripts downloaded from GitHub at runtime
- NVMe hostPath storage for models
- emptyDir for temporary results
- Kueue integration for queue management

## Default Resources

- CPU: 8 cores
- Memory: 128Gi
- GPU: 1x A100

Override with environment variables:
```bash
NS=eidf230ns CPU=16 MEMORY=256Gi GPU=2 k8s/deploy-benchmark.sh
```
