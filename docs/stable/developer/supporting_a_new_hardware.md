---
sidebar_position: 0
---

# Supporting a New Hardware

ServerlessLLM actively expands support for new hardware configurations to meet diverse deployment needs.

## Support Standards
Hardware is considered supported by ServerlessLLM if:
1. Any of the inference backends used (e.g., Transformers, vLLM) can run model inference on the hardware.
2. ServerlessLLM Store can successfully load model checkpoints on the hardware.

## Steps to Support a New Hardware
1. **Check Inference Backend Compatibility**: Refer to the specific inference backend documentation (e.g., for vLLM, Transformers) for hardware support.
2. **ServerlessLLM Store Configuration**:
   - If the hardware provides CUDA-compatible APIs (e.g., ROCm), adjust the build script (`CMakeLists.txt`) by adding necessary compiler flags.
   - For non-CUDA-compatible APIs, implementing a custom checkpoint loading function might be required.

## Verifying Hardware Support in ServerlessLLM Store
The hardware support is verified if it successfully completes the [Quick Start Guide](https://serverlessllm.github.io/docs/stable/getting_started/quickstart/) examples, showcasing checkpoint loading and inference functionality without errors.

If you encounter any issues or have questions, please reach out to the ServerlessLLM team by raising an issue on the [GitHub repository](https://github.com/ServerlessLLM/ServerlessLLM/issues).