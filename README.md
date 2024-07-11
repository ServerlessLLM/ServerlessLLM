<p align="center">
  <img src="./docs/images/serverlessllm.jpg" alt="ServerlessLLM Logo" width="30%">
</p>

# ServerlessLLM

ServerlessLLM is a fast, cost-effective and easy-to-use library designed for multi-model serving, also known as [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html), [Inference Endpoint](https://huggingface.co/inference-endpoints/dedicated), or [Model Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2). This library is ideal for environments with limited GPU resources (GPU poor), as it allows efficient dynamic loading of models onto GPUs. By supporting high levels of GPU multiplexing, it maximizes GPU utilization without the need to dedicate GPUs to individual models.

## News

- We are working towards to the first release and making  documentation ready. Stay tuned!

## About

ServerlessLLM is Fast:

- Supports various leading LLM inference libraries including [vLLM](https://github.com/vllm-project/vllm) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/index).
- Capable of loading 70B LLMs (checkpoint size beyond 100GB) onto an 8-GPU server in just 300-700ms, achieving load times 5-10X faster than [Safetensors](https://github.com/huggingface/safetensors) and PyTorch Checkpoint Loader.
- Support start-time-optimized model loading scheduler, achieving 5-100X better LLM start-up latency than [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and [KServe](https://github.com/kserve/kserve).

ServerlessLLM is Cost-Effective:

- Supports many LLM models to share a few GPUs with low model switching overhead and seamless inference live migration.
- Fully utilizes local storage resources available on multi-GPU servers, reducing the need for employing costly storage servers and network bandwidth.

ServerlessLLM is Easy-to-Use:

- Facilitates easy deployment via [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) and [Kubernetes](https://kubernetes.io/) (comming soon).
- Seamlessly deploys [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/index) models and your custom LLM models.
- Integrates seamlessly with the [OpenAI Query API](https://platform.openai.com/docs/introduction).

## Getting Started

1. Install ServerlessLLM following [Installation Guide](https://serverlessllm.github.io/docs/stable/getting_started/installation/).

2. Start a local ServerlessLLM cluster:

```bash
sllm-serve start
```

3. Deploy a model using the ServerlessLLM CLI:

```bash
sllm-cli deploy --model facebook/opt-1.3b
```

4. Run an LLM inference following [documentation](https://serverlessllm.github.io/):

```bash
sllm-cli generate --model facebook/opt-1.3b --input input.json
```

## Performance

A detailed analysis of the performance of ServerlessLLM is [here](./benchmarks/README.md).

## Contributing

ServerlessLLM is actively maintained and developed by those [Contributors](./CONTRIBUTING.md). We welcome new contributors to join us in making ServerlessLLM faster, better and more easier to use. Please check [Contributing Guide](./CONTRIBUTING.md) for details.

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
