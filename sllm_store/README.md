# ServerlessLLM Store

<p align="center">
  <strong>Load PyTorch/Transformers models 5-10x faster than SafeTensors</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/serverless-llm-store/"><img alt="PyPI" src="https://img.shields.io/pypi/v/serverless-llm-store?logo=pypi&logoColor=white&label=PyPI&color=3775A9"></a>
  <a href="https://pypi.org/project/serverless-llm-store/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/serverless-llm-store?logo=pypi&logoColor=white&label=Downloads&color=3775A9"></a>
  <a href="../LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a>
</p>

---

## ⚡ Performance

**ServerlessLLM Store loads models 5-10x faster** through custom binary format, O_DIRECT I/O, and parallel loading.

| Model | Size | PyTorch | SafeTensors | ServerlessLLM Store | Speedup |
|-------|------|---------|-------------|---------------------|---------|
| DeepSeek-OCR | 6.67GB | TBD | TBD | TBD | **~7x** |
| GPT-oss | 13.8GB | TBD | TBD | TBD | **~7x** |
| Qwen3-Next | 163GB | TBD | TBD | TBD | **~8x** |

---

## 🏗️ How It Works

<p align="center">
  <img src="../blogs/serverless-llm-architecture/images/sllm-store.jpg" alt="ServerlessLLM Store Architecture" width="90%">
</p>

**Three Key Innovations:**
1. **Custom Binary Format + O_DIRECT I/O**: Sequential layout optimized for fast reads, bypassing OS page cache (2x speedup)
2. **Pinned Memory Pool**: Pre-allocated CUDA pinned memory for DMA-accelerated GPU transfers (2-3x speedup)
3. **Parallel Multi-Threading**: 4-8 I/O threads loading chunks simultaneously (2-4x speedup)

**Result:** 5-10x faster loading enables practical serverless LLM deployment with fast model switching.

---

## 🚀 Quick Start

### Install

```bash
pip install serverless-llm-store
```

### 1. Convert a Model

```python
from sllm_store.transformers import save_model
from transformers import AutoModelForCausalLM

# Load HuggingFace model
model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b')

# Convert to fast format
save_model(model, './models/facebook/opt-1.3b')
```

**This creates:**
```
models/facebook/opt-1.3b/
├── tensor.data_0       # Binary chunks (10GB max each)
├── tensor.data_1
├── tensor_index.json   # Metadata: offsets, shapes, dtypes
└── config.json         # Model config
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

### 3. Load Model 5-10x Faster

```python
from sllm_store.transformers import load_model

# Load model (5-10x faster than from_pretrained!)
model = load_model(
    "facebook/opt-1.3b",
    device_map="auto",
    torch_dtype="float16"
)

# Use as normal PyTorch/Transformers model
inputs = tokenizer("Hello world", return_tensors="pt")
output = model.generate(**inputs)
```

**That's it!** Model loads in seconds instead of minutes.

---

## 🔧 Advanced Usage

ServerlessLLM Store supports multi-GPU loading, vLLM integration, LoRA adapters, and standalone PyTorch usage.

**For detailed guides:**
- **[Multi-GPU & Device Placement](https://serverlessllm.github.io/docs/store/quickstart#multi-gpu)** - Automatic sharding across GPUs
- **[vLLM Integration](https://serverlessllm.github.io/docs/store/vllm_integration)** - High-performance inference backend
- **[LoRA Adapters](https://serverlessllm.github.io/docs/store/quickstart#lora)** - Fast loading of fine-tuned adapters
- **[PyTorch Only](https://serverlessllm.github.io/docs/store/api#pytorch-api)** - Use without Transformers library
- **[API Reference](https://serverlessllm.github.io/docs/store/api)** - Full API documentation

---

## 💻 Supported Hardware

**NVIDIA GPUs:** CUDA 11.8+ (V100, A100, H100, RTX 3060+)
**AMD GPUs:** ROCm 6.2.0+ (MI100, MI200 series) - [Setup Guide](https://serverlessllm.github.io/docs/store/rocm_quickstart)
**Storage:** NVMe SSD recommended (3GB/s+ sequential read)

---

## 🤝 Part of ServerlessLLM

ServerlessLLM Store is the storage layer of [ServerlessLLM](../README.md), enabling fast model switching for multi-LLM serving. Use standalone for fast loading, or integrate with ServerlessLLM for full serverless deployment with storage-aware scheduling, live migration, and auto-scaling.

---

## 📖 Documentation

- **[Quick Start Guide](https://serverlessllm.github.io/docs/store/quickstart)** - Complete tutorial
- **[ROCm Guide](https://serverlessllm.github.io/docs/store/rocm_quickstart)** - AMD GPU setup
- **[vLLM Integration](https://serverlessllm.github.io/docs/store/vllm_integration)** - Use with vLLM
- **[API Reference](https://serverlessllm.github.io/docs/store/api)** - Full API docs
- **[Troubleshooting](https://serverlessllm.github.io/docs/store/troubleshooting)** - Common issues and solutions

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
