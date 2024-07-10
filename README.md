# ServerlessLLM

ServerlessLLM is a fast, cost-effective and easy-to-use library designed for multi-model serving, also known as [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html), [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated), or [Model Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2). This library is ideal for environments with limited GPU resources (GPU poor), as it allows dynamic loading of models onto GPUs. By supporting high levels of GPU multiplexing, it maximizes GPU utilization without the need to dedicate GPUs to individual models.

## About

ServerlessLLM is Fast:

- Supports various leading LLM inference libraries including [vLLM](https://github.com/vllm-project/vllm) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/index).
- Capable of loading 70B LLMs (checkpoint size beyond 100GB) onto an 8-GPU server in just 700ms, achieving load times 5 to 10 times faster than [Safetensors](https://github.com/huggingface/safetensors) and PyTorch Checkpoint Loader.
- Support locality-aware GPU cluster scheduler and LLM infernece live migration, contributing to lower mean and P99 first-token-latency than [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and [KServe](https://github.com/kserve/kserve).

ServerlessLLM is Cost-Effective:

- Enables hundreds of LLM models to share a few GPUs with minimal model switching overhead.
- Utilizes local storage resources on multi-GPU servers efficiently, eliminating the need for additional costly storage and network bandwidth.

ServerlessLLM is Easy-to-Use:

- Facilitates easy deployment via [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) and [Kubernetes](https://kubernetes.io/) (comming soon).
- Deploys HuggingFace Transformers models with a single command.
- Integrates seamlessly with the OpenAI query API.

## Getting Started

1. Install ServerlessLLM:

Detailed installation guide: [ServerlessLLM Installation](./docs/installation.md).

2. Start a local ServerlessLLM cluster:

```bash
sllm-serve start
```

3. Deploy a model using the ServerlessLLM CLI:

```bash
sllm-cli deploy --model facebook/opt-1.3b
```

4. Run an LLM inference for the deployed model:

```bash
sllm-cli generate --model facebook/opt-1.3b --input input.json
```

Detailed usage guide: [ServerlessLLM Usage APIs](https://serverlessllm.github.io/).

## Performance

A detailed analysis of the performance of ServerlessLLM is [here](./benchmarks/README.md).

## Contributing

We are actively developing ServerlessLLM and we plan to realize a roadmap soon. We also highly welcome contributors to ServerlessLLM and will acknowledge your contributions on the main page. Please check [Contributing Guide](./CONTRIBUTING.md) for details.

## Citation

If you use ServerlessLLM for your research, please cite our [paper](https://arxiv.org/abs/2401.14351):

```bibtex
@article{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={USENIX Symposium on Operating Systems Design and Implementation (OSDI'24)},
  year={2024}
}
```
