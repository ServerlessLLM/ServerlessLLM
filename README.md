<p align="center">
  <picture>
    <img src="./docs/images/serverlessllm.jpg" alt="ServerlessLLM" width="30%">
  </picture>
</p>

<p align="center">
| <a href="https://serverlessllm.github.io"><b>Documentation</b></a> | <a href="https://www.usenix.org/conference/osdi24/presentation/fu"><b>Paper</b></a> | <a href="https://discord.gg/AEF8Gduvm8"><b>Discord</b></a> | <a href="./docs/images/wechat.png"><b>WeChat</b></a> |

</p>

# ServerlessLLM

ServerlessLLM (`sllm`, pronounced "slim") is an open-source serverless framework designed to make custom and elastic LLM deployment easy, fast, and affordable. As LLMs grow in size and complexity, deploying them on AI hardware has become increasingly costly and technically challenging, limiting custom LLM deployment to only a select few. ServerlessLLM solves these challenges with a full-stack, LLM-centric serverless system design, optimizing everything from checkpoint formats and inference runtimes to the storage layer and cluster scheduler.

Curious about how it works under the hood? Check out our [System Walkthrough](https://github.com/ServerlessLLM/ServerlessLLM/tree/main/blogs/serverless-llm-architecture) for a deep dive into the technical design—perfect if you're exploring your own research or building with ServerlessLLM.

## News

- **[03/25]** We're excited to share that we'll be giving a ServerlessLLM tutorial at the SESAME workshop, co-located with ASPLOS/EuroSys 2025 in Rotterdam, Netherlands, on March 31. [More info](https://sesame25.github.io/)
- **[11/24]** We have added experimental support of fast checkpoint loading for AMD GPUs (ROCm) when using with vLLM, PyTorch and HuggingFace Accelerate. Please refer to the [documentation](https://serverlessllm.github.io/docs/stable/store/rocm_quickstart) for more details.
- **[10/24]** ServerlessLLM was invited to present at a global AI tech vision forum in Singapore.
- **[10/24]** We hosted the first ServerlessLLM developer meetup in Edinburgh, attracting over 50 attendees both offline and online. Together, we brainstormed many exciting new features to develop. If you have great ideas, we’d love for you to join us!
- **[10/24]** We made the first public release of ServerlessLLM. Check out the details of the release [here](https://github.com/ServerlessLLM/ServerlessLLM/releases/tag/v0.5.0).
- **[09/24]** ServerlessLLM now supports embedding-based RAG + LLM deployment. We’re preparing a blog and demo—stay tuned!
- **[08/24]** ServerlessLLM added support for vLLM.
- **[07/24]** We presented ServerlessLLM at Nvidia's headquarters.
- **[06/24]** ServerlessLLM officially went public.

## Goals

ServerlessLLM is designed to support multiple LLMs in efficiently sharing limited AI hardware and dynamically switching between them on demand, which can increase hardware utilization and reduce the cost of LLM services. This multi-LLM scenario, commonly referred to as Serverless, is highly sought after by AI practitioners, as seen in solutions like [Serverless Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html), [Inference Endpoints](https://huggingface.co/inference-endpoints/dedicated), and [Model Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints?view=azureml-api-2). However, these existing offerings often face performance overhead and scalability challenges, which ServerlessLLM effectively addresses through three key capabilities:

**ServerlessLLM is Fast**:
- Supports leading LLM inference libraries like [vLLM](https://github.com/vllm-project/vllm) and [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/index). Through vLLM, ServerlessLLM can support various types of AI hardware (summarized by vLLM at [here](https://docs.vllm.ai/en/stable/getting_started/installation.html))
- Achieves 5-10X faster loading speeds compared to [Safetensors](https://github.com/huggingface/safetensors) and the PyTorch Checkpoint Loader.
- Features an optimized model loading scheduler, offering 5-100X lower start-up latency than [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) and [KServe](https://github.com/kserve/kserve).

**ServerlessLLM is Cost-Efficient**:
- Allows multiple LLM models to share GPUs with minimal model switching overhead and supports seamless inference live migration.
- Maximizes the use of local storage on multi-GPU servers, reducing the need for expensive storage servers and excessive network bandwidth.

**ServerlessLLM is Easy-to-Use**:
- Simplifies deployment through [Ray Cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) and [Kubernetes](https://kubernetes.io/) via [KubeRay](https://github.com/ray-project/kuberay).
- Supports seamless deployment of [HuggingFace Transformers](https://huggingface.co/docs/transformers/en/index) and custom LLM models.
- Supports NVIDIA and AMD GPUs
- Easily integrates with the [OpenAI Query API](https://platform.openai.com/docs/overview).

## Getting Started

1. Install ServerlessLLM with pip or [from source](https://serverlessllm.github.io/docs/stable/getting_started/installation/).

```bash
# On the head node
conda create -n sllm python=3.10 -y
conda activate sllm
pip install serverless-llm

# On a worker node
conda create -n sllm-worker python=3.10 -y
conda activate sllm-worker
pip install serverless-llm[worker]
```

2. Start a local ServerlessLLM cluster using the [Quick Start Guide](https://serverlessllm.github.io/docs/stable/getting_started/quickstart/).

3. Want to try fast checkpoint loading in your own code? Check out the [ServerlessLLM Store Guide](https://serverlessllm.github.io/docs/stable/store/quickstart).

## Documentation

To install ServerlessLLM, please follow the steps outlined in our [documentation](https://serverlessllm.github.io). ServerlessLLM also offers Python APIs for loading and unloading checkpoints, as well as CLI tools to launch an LLM cluster. Both the CLI tools and APIs are demonstrated in the documentation.

## Benchmark

Benchmark results for ServerlessLLM can be found [here](./benchmarks/README.md).

## Community

ServerlessLLM is maintained by a global team of over 10 developers, and this number is growing. If you're interested in learning more or getting involved, we invite you to join our community on [Discord](https://discord.gg/AEF8Gduvm8) and [WeChat](./docs/images/wechat.png). Share your ideas, ask questions, and contribute to the development of ServerlessLLM. For becoming a contributor, please refer to our [Contributor Guide](./CONTRIBUTING.md).

## Citation

If you use ServerlessLLM for your research, please cite our [paper](https://arxiv.org/abs/2401.14351):

```bibtex
@inproceedings{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24)},
  pages={135--153},
  year={2024}
}
```
