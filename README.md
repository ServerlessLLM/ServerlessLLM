# ServerlessLLM

Easy, Cheap and Fast serverless inference for LLMs.

## About

ServerlessLLM exploits the substantial capacity and bandwidth of storage and memory available on GPU servers, thereby reducing costly remote checkpoint downloads and achieving fast serverless cold-start. 

**Fast:**

- Fast start-up: second-level model loading

- High performance inference: support vLLM

**Easy to deploy**

- Easy to deploy: ray and k8s.
- OpenAI compatible: support OpenAI API
- Ray ecosystems: support Ray Serve API
- Future support for AMD GPU

**Module**

- Checkpoint store

## Easy to extend

- Any inference engine.
- And policy.
- 
**Cheap:**

- No dedicated GPU: launch as you use

- Time-sharing GPU: enable deploy multi-model workflow with only one GPU

**Serverless deployment**

- Global checkpoint management: checkpoint placement policy.

- Start-up time optimzied scheduler.

- LLM inference with live migration.

## Performance

ServerlessLLM achieves xxX speedup in model loading time compared to Safetensors and xxX in time-to-first-token compared to Ray Serve. Please find detailed performance evaluation in [Benchmarks](./benchmarks/README.md).

## Getting Started

1. Install the ServerlessLLM library:

```bash
pip install serverless_llm
```

More detailed installation instructions can be found in the [Installation Guide](./docs/installation.md).

2. Start a local ServerlessLLM cluster:

```bash
sllm-serve start
```

3. Deploy a model using the ServerlessLLM CLI:

```bash
sllm-cli deploy --model facebook/opt-1.3b
```

4. Make a generation using the deployed model:

```bash
sllm-cli generate --model facebook/opt-1.3b --input input.json
```

More detailed usage instructions can be found in the [Usage Guide](https://serverlessllm.github.io/).


## Contributing

We welcome contributions to ServerlessLLM! Please refer to our [Contributing Guide](./CONTRIBUTING.md) for more details.

## Citation

If you use ServerlessLLM for your research, please cite our [paper](https://arxiv.org/abs/2401.14351):
```bibtex
@article{fu2024serverlessllm,
  title={ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  journal={arXiv preprint arXiv:2401.14351},
  year={2024}
}
```
