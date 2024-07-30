---
sidebar_position: 1
---

# Serverless LLM

<!-- logo (../images/serverlessllm.jpg) -->
![ServerlessLLM](../images/serverlessllm.jpg)

ServerlessLLM is a **fast** and **easy-to-use** serving system designed for **affordable** multi-LLM serving, also known as LLM-as-a-Service. ServerlessLLM is ideal for environments with many LLMs to serve on limited GPU resources, as it enables efficient dynamic loading of LLMs onto GPUs. By elasticly scaling model instances and multiplexing GPUs, ServerlessLLM can significantly reduce costs compared to traditional GPU-dedicated serving systems while still providing low-latency (Time-to-First-Token, TTFT) LLM completions.

## Documentation

### Getting Started

- [Install ServerlessLLM](./getting_started/installation.md)
- [Deploy a ServerlessLLM cluster on your local machine](./getting_started/quickstart.md)
- [Deploy ServerlessLLM using Docker (Recommended)](./getting_started/docker_quickstart.md)
- [Deploy ServerlessLLM on a multi-machine cluster](./getting_started/multi_machine_setup.md)

### ServerlessLLM Store

- [Use ServerlessLLM Store in your own code](./store/quickstart.md)

### ServerlessLLM CLI

- [ServerlessLLM CLI Documentation](./cli/cli_api.md)
