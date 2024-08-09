---
sidebar_position: 1
---

# Serverless LLM

<!-- Scaled logo -->
<img src="../images/serverlessllm.jpg" alt="ServerlessLLM" width="256px">

ServerlessLLM is a **fast** and **easy-to-use** serving system designed for **affordable** multi-LLM serving, also known as LLM-as-a-Service. ServerlessLLM is ideal for environments with many LLMs to serve on limited GPU resources, as it enables efficient dynamic loading of LLMs onto GPUs. By elasticly scaling model instances and multiplexing GPUs, ServerlessLLM can significantly reduce costs compared to traditional GPU-dedicated serving systems while still providing low-latency (Time-to-First-Token, TTFT) LLM completions.

## Documentation

### Getting Started

- [Install ServerlessLLM](./getting_started/installation.md)
- [Quickstart](./getting_started/quickstart.md)
- [Quickstart with Docker](./getting_started/docker_quickstart.md)
- [Multi-machine quickstart](./getting_started/multi_machine_setup.md)

### ServerlessLLM Serve

- [Storage-aware scheduler](./serve/storage_aware_scheduling.md)

### ServerlessLLM Store

- [Quickstart](./store/quickstart.md)

### ServerlessLLM CLI

- [ServerlessLLM CLI API](./cli/cli_api.md)
