---
sidebar_position: 0
---

# Serverless LLM

<!-- Scaled logo -->
<img src={require('./images/serverlessllm.jpg').default} alt="ServerlessLLM" width="256px"/>

ServerlessLLM is a **fast** and **easy-to-use** serving system designed for **affordable** multi-LLM serving, also known as LLM-as-a-Service. ServerlessLLM is ideal for environments with multiple LLMs that need to be served on limited GPU resources, as it enables efficient dynamic loading of LLMs onto GPUs. By elastically scaling model instances and multiplexing GPUs, ServerlessLLM can significantly reduce costs compared to traditional GPU-dedicated serving systems while still providing low-latency (Time-to-First-Token, TTFT) LLM completions.

ServerlessLLM now supports NVIDIA and AMD GPUs, including following hardware:
* NVIDIA GPUs: Compute Capability 7.0+ (e.g, V100, A100, RTX A6000, GeForce RTX 3060)
* AMD GPUs: ROCm 6.2.0+ (tested on MI100s and MI200s)

## Documentation

### Getting Started

- [Quickstart](./getting_started.md)
- [Single Machine Deployment (From Scratch)](./deployment/single_machine.md)
- [Multi-machine Deployment](./deployment/multi_machine.md)
- [SLURM Cluster Deployment](./deployment/slurm_cluster.md)

### Advanced Features

- [Storage-Aware Scheduler](./features/storage_aware_scheduling.md)
- [Live Migration](./features/live_migration.md)
- [PEFT LoRA Serving](./features/peft_lora_serving.md)

### ServerlessLLM Store

- [Quickstart](./store/quickstart.md)
- [ROCm Quickstart](./store/rocm_quickstart.md)

### ServerlessLLM CLI

- [ServerlessLLM CLI API](./api/cli.md)
