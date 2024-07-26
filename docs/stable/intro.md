---
sidebar_position: 1
---

# Serverless LLM

ServerlessLLM is a fast, affordable, and easy-to-use library designed for multi-LLM serving, also known as [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html), [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated), or [Model Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2). This library is ideal for environments with limited GPU resources, as it allows efficient dynamic loading of models onto GPUs. By supporting high levels of GPU multiplexing, it maximizes GPU utilization without the need to dedicate GPUs to individual models.

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
