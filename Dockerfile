# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
ARG CUDA_VERSION=12.8.1
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder
ARG CUDA_VERSION=12.8.1
ARG PYTHON_VERSION=3.10
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
ENV UV_LINK_MODE=copy
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create build venv with uv
RUN uv venv /opt/build --python 3.10
ENV VIRTUAL_ENV=/opt/build
ENV PATH="/opt/build/bin:$PATH"

# Set the working directory
WORKDIR /app

# Build checkpoint store
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
COPY sllm_store/requirements-build.txt /app/sllm_store/requirements-build.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /app/sllm_store/requirements-build.txt setuptools wheel

COPY sllm_store/cmake /app/sllm_store/cmake
COPY sllm_store/CMakeLists.txt /app/sllm_store/CMakeLists.txt
COPY sllm_store/csrc /app/sllm_store/csrc
COPY sllm_store/sllm_store /app/sllm_store/sllm_store
COPY sllm_store/setup.py /app/sllm_store/setup.py
COPY sllm_store/pyproject.toml /app/sllm_store/pyproject.toml
COPY sllm_store/MANIFEST.in /app/sllm_store/MANIFEST.in
COPY sllm_store/requirements.txt /app/sllm_store/requirements.txt
COPY sllm_store/README.md /app/sllm_store/README.md
COPY sllm_store/proto/storage.proto /app/sllm_store/proto/storage.proto
RUN cd sllm_store && python setup.py bdist_wheel


# Copy only dependencies and build config first (for caching)
COPY requirements.txt requirements-vllm.txt /app/
COPY pyproject.toml setup.py py.typed /app/
COPY sllm/backends /app/sllm/backends
COPY sllm/ft_backends /app/sllm/ft_backends
COPY sllm/cli /app/sllm/cli
COPY sllm/*.py /app/sllm/
COPY README.md /app/
RUN python setup.py bdist_wheel

#################### RUNTIME IMAGE ####################
# Using nvidia/cuda devel base:
# - Has nvcc (needed by SGLang's flashinfer JIT during CUDA graph capture)
# - Each venv installs its own torch version (vLLM/SGLang need 2.9.x)
# - Cleaner dependency management (no unused pytorch base)
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

# Set environment for v1-beta architecture
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STORAGE_PATH=/models \
    HEAD_HOST=0.0.0.0 \
    HEAD_PORT=8343 \
    UV_LINK_MODE=copy

# Install runtime dependencies + uv
RUN apt-get update -y && \
    apt-get install -y curl netcat-openbsd gcc g++ libnuma1 git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set the working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/
COPY requirements-vllm.txt /app/
COPY sllm_store/requirements.txt /app/requirements-sllm-store.txt

# Create head venv (no system-site-packages, isolated control plane)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venvs/head --python 3.10 && \
    uv pip install --python /opt/venvs/head/bin/python -r /app/requirements.txt

# Create isolated venvs for pylet worker architecture
# Each venv has its own dependencies (no system-site-packages)

# pylet venv: minimal - just pylet for spawning processes
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venvs/pylet --python 3.10 && \
    uv pip install --python /opt/venvs/pylet/bin/python "pylet>=0.4.0"

# sllm-store venv: sllm-store with torch, transformers, grpc
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venvs/sllm-store --python 3.10 && \
    uv pip install --python /opt/venvs/sllm-store/bin/python \
        -r /app/requirements-sllm-store.txt

# vllm venv: vLLM inference backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venvs/vllm --python 3.10 && \
    uv pip install --python /opt/venvs/vllm/bin/python \
        -r /app/requirements-vllm.txt

# sglang venv: SGLang inference backend
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venvs/sglang --python 3.10 && \
    uv pip install --python /opt/venvs/sglang/bin/python sglang --prerelease=allow

# Copy vLLM patch
COPY sllm_store/vllm_patch /app/vllm_patch

# Copy examples folder
COPY examples /app/examples

# Copy the built wheels from the builder
COPY --from=builder /app/sllm_store/dist /app/sllm_store/dist
COPY --from=builder /app/dist /app/dist

# Install ServerlessLLM packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venvs/head/bin/python \
        /app/sllm_store/dist/*.whl /app/dist/*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venvs/sllm-store/bin/python \
        /app/sllm_store/dist/*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venvs/vllm/bin/python \
        /app/sllm_store/dist/*.whl /app/dist/*.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venvs/sglang/bin/python \
        /app/sllm_store/dist/*.whl /app/dist/*.whl

# Install MoE-CAP for expert distribution tracking
RUN git clone https://github.com/Auto-CAP/MoE-CAP.git /tmp/MoE-CAP

# Install MoE-CAP in editable mode in each venv
RUN bash -c "source /opt/venvs/head/bin/activate && cd /tmp/MoE-CAP && uv pip install -e ."

RUN bash -c "source /opt/venvs/vllm/bin/activate && cd /tmp/MoE-CAP && uv pip install -e ."

RUN bash -c "source /opt/venvs/sglang/bin/activate && cd /tmp/MoE-CAP && uv pip install -e ."
# Apply vLLM patch in vllm venv
RUN bash -c "source /opt/venvs/vllm/bin/activate && cd /app && ./vllm_patch/patch.sh"

# Copy the entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Add head venv to PATH for sllm CLI
ENV PATH="/opt/venvs/head/bin:$PATH"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD /app/entrypoint.sh health

# Expose ports
# Head node: 8343 (API Gateway)
# Pylet head: 8000 (cluster manager)
# vLLM instances: 8080-8179 (spawned by Pylet)
EXPOSE 8343 8000 8080-8179

# Labels for container identification
LABEL org.opencontainers.image.title="ServerlessLLM v1-beta" \
      org.opencontainers.image.description="Pylet-based distributed LLM serving platform" \
      org.opencontainers.image.version="1.0.0-beta" \
      org.opencontainers.image.vendor="ServerlessLLM Team" \
      org.opencontainers.image.licenses="Apache-2.0"

# Set the entrypoint directly to the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
