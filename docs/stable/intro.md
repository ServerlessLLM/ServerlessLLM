---
sidebar_position: 1
---

# Serverless LLM

<!-- Scaled logo -->
<img src="/img/serverlessllm.jpg" alt="ServerlessLLM" width="256px"/>

ServerlessLLM is a **fast** and **easy-to-use** serving system designed for **affordable** multi-LLM serving, also known as LLM-as-a-Service. ServerlessLLM is ideal for environments with multiple LLMs that need to be served on limited GPU resources, as it enables efficient dynamic loading of LLMs onto GPUs. By elastically scaling model instances and multiplexing GPUs, ServerlessLLM can significantly reduce costs compared to traditional GPU-dedicated serving systems while still providing low-latency (Time-to-First-Token, TTFT) LLM completions.

ServerlessLLM now supports NVIDIA and AMD GPUs, we have tested ServerlessLLM on the following hardwares:
* NVIDIA GPUs: Compute Capability 8.0+ (e.g, A100, RTX A5000, RTX A6000)
* AMD GPUs: ROCm 6.2.0+

## Documentation

### Getting Started

- [Install ServerlessLLM](./getting_started/installation.md)
- [Quickstart](./getting_started/quickstart.md)
- [Quickstart with Docker](./getting_started/docker_quickstart.md)
- [Multi-machine Quickstart](./getting_started/multi_machine_setup.md)

### ServerlessLLM Serve

- [Storage-Aware Scheduler](./serve/storage_aware_scheduling.md)

### ServerlessLLM Store

- [Quickstart](./store/quickstart.md)
- [ROCm Installation(Experimental)](./store/installation_with_rocm.md)

### ServerlessLLM CLI

- [ServerlessLLM CLI API](./cli/cli_api.md)
