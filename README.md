# ServerlessLLM

ServerlessLLM is a locality-enhanced serverless inference system for Large Language Models (LLMs). 
ServerlessLLM exploits the substantial capacity and bandwidth of storage and memory devices available on GPU servers, thereby reducing costly remote checkpoint downloads and achieving efficient checkpoint loading. 
ServerlessLLM achieves this through three main contributions: 
(i) fast LLM checkpoint loading via a novel loading-optimized checkpoint format design, coupled with an efficient multi-tier checkpoint loading system; 
(ii) locality-driven LLM inference with live migration, which allows ServerlessLLM to effectively achieve locality-driven server allocation while preserving the low latency of ongoing LLM inference; 
and (iii) locality-aware server allocation, enabling ServerlessLLM to evaluate the status of each server in a cluster and effectively schedule model startup time to capitalize on local checkpoint placement. 
Our comprehensive experiments, which include microbenchmarks and real-world traces, show that ServerlessLLM surpasses state-of-the-art systems by 10 - 200X in latency performance when running various LLM inference workloads.

For more details, please refer to our paper: [ServerlessLLM: Locality-Enhanced Serverless Inference for Large Language Models](https://arxiv.org/abs/2401.14351).

## Roadmap

This roadmap provides an overview of our current plans and the direction we intend to take the project. Please note that this roadmap is subject to change based on community feedback, market changes, and development resource availability.

### [FasterCkpt] Fast Checkpoint Loading
- **Multi-Tier Checkpoint Loading**

_Expected Release Date: February, 2024_

### [LLMServe] Locality-Enhanced Serverless Inference
- **Locality-Driven LLM Inference**
- **Locality-Aware Server Allocation**

_Expected Release Date: April, 2024_

### Feedback
Your feedback and suggestions are important to us. Feel free to open an issue or propose changes to help us improve.

_Note: The roadmap is for informational purposes and is not a commitment, promise, or legal obligation to deliver any material, code, or functionality. The development, release, and timing of any features or functionality remain at the sole discretion of the project maintainers._
