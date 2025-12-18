# ServerlessLLM Store

<p align="center">
  <strong>Load PyTorch/Transformers models 6-10x faster than SOTA checkpoint loaders</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/serverless-llm-store/"><img alt="PyPI" src="https://img.shields.io/pypi/v/serverless-llm-store?logo=pypi&logoColor=white&label=PyPI&color=3775A9"></a>
  <a href="https://pypi.org/project/serverless-llm-store/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/serverless-llm-store?logo=pypi&logoColor=white&label=Downloads&color=3775A9"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
</p>

---

## ⚡ Performance

**ServerlessLLM Store loads models 6-10x faster** through custom binary format, O_DIRECT I/O, and parallel loading.

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Scenario</th>
      <th>SafeTensors</th>
      <th>ServerlessLLM Store</th>
      <th>Speedup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen/Qwen3-32B</td>
      <td>Random</td>
      <td>20.6s</td>
      <td>3.2s</td>
      <td><strong>6.40x</strong></td>
    </tr>
    <tr>
      <td>Cached</td>
      <td>12.5s</td>
      <td>1.3s</td>
      <td><strong>9.95x</strong></td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-R1-Distill-Qwen-32B</td>
      <td>Random</td>
      <td>19.1s</td>
      <td>3.2s</td>
      <td><strong>5.93x</strong></td>
    </tr>
    <tr>
      <td>Cached</td>
      <td>10.2s</td>
      <td>1.2s</td>
      <td><strong>8.58x</strong></td>
    </tr>
    <tr>
      <td>Llama-3.1-8B-Instruct</td>
      <td>Random</td>
      <td>4.4s</td>
      <td>0.7s</td>
      <td><strong>6.54x</strong></td>
    </tr>
  </tbody>
</table>

*Results obtained on NVIDIA H100 GPUs with NVMe SSD. "Random" simulates serverless multi-model serving; "Cached" shows repeated loading of the same model.*

---

## 🏗️ How It Works

<p align="center">
  <img src="../blogs/serverless-llm-architecture/images/sllm-store.jpg" alt="ServerlessLLM Store Architecture" width="90%">
</p>

**Three Key Innovations:**
1. **Custom Binary Format + O_DIRECT I/O**: Sequential layout optimized for fast reads, bypassing OS page cache (2x speedup)
2. **Pinned Memory Pool**: Pre-allocated CUDA pinned memory for DMA-accelerated GPU transfers (2-3x speedup)
3. **Parallel Multi-Threading**: 4-8 I/O threads loading chunks simultaneously (2-4x speedup)

**Result:** 6-10x faster loading enables practical serverless LLM deployment with fast model switching.

---

## 🚀 Quick Start

### Install

```bash
pip install serverless-llm-store
```

### 1. Convert a Model

```bash
sllm-store save --model Qwen/Qwen3-0.6B --backend transformers
```

### 2. Start Store Server

```bash
sllm-store start \
  --storage-path ./models \
  --mem-pool-size 4GB \
  --num-threads 4
```

**Parameters:**
- `--storage-path`: Where converted models are stored
- `--mem-pool-size`: Pinned memory pool size (must be ≥ largest model)
- `--num-threads`: I/O threads for parallel loading (4-8 recommended)

### 3. Load Model 6-10x Faster

```python
from sllm_store.transformers import load_model

# Load model (6-10x faster than from_pretrained!)
model = load_model(
    "Qwen/Qwen3-0.6B",
    device_map="auto",
    torch_dtype="float16"
)

# Use as normal PyTorch/Transformers model
inputs = tokenizer("Hello world", return_tensors="pt")
output = model.generate(**inputs)
```

**That's it!** Model loads in a second!

---

## 🔧 Advanced Usage

ServerlessLLM Store supports quantization, vLLM integration, LoRA adapters, and standalone PyTorch usage.

**For detailed guides:**
- **[Quantization](https://serverlessllm.github.io/docs/store/quantization)** - Quantize models during loading
- **[vLLM Integration](https://serverlessllm.github.io/docs/store/quickstart#usage-with-vllm)** - Use with vLLM
- **[LoRA Adapters](https://serverlessllm.github.io/docs/store/quickstart#usage-examples-1a)** - Fast loading of fine-tuned adapters
- **[CLI API Reference](https://serverlessllm.github.io/docs/api/sllm-store-cli)** - CLI API documentation

---

## 💻 Supported Hardware

**NVIDIA GPUs:** Compute capability 7.0+ (V100, A100, H100, RTX 3060+)
**AMD GPUs:** ROCm 6.2+ (MI100, MI200 series) - Experimental

---

## 🤝 Part of ServerlessLLM

ServerlessLLM Store is the storage layer of [ServerlessLLM](../README.md), enabling fast model switching for multi-LLM serving. Use standalone for fast loading, or integrate with ServerlessLLM for full serverless deployment with storage-aware scheduling, live migration, and auto-scaling.

---

## 📄 Citation

If you use ServerlessLLM Store in your research, please cite our [OSDI'24 paper](https://www.usenix.org/conference/osdi24/presentation/fu):

```bibtex
@inproceedings{fu2024serverlessllm,
  title={ServerlessLLM: Low-Latency Serverless Inference for Large Language Models},
  author={Fu, Yao and Xue, Leyang and Huang, Yeqi and Brabete, Andrei-Octavian and Ustiugov, Dmitrii and Patel, Yuvraj and Mai, Luo},
  booktitle={OSDI'24},
  year={2024}
}
```

---

## 📝 License

Apache 2.0 - See [LICENSE](../LICENSE)

---

<p align="center">
  <strong>⭐ Star this repo if ServerlessLLM Store helps you!</strong>
</p>
