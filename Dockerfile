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

# HTTP-based ServerlessLLM Architecture (No Ray dependencies)
ARG CUDA_VERSION=12.1.1
#################### BASE BUILD IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder
ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.10
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo wget \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Set the working directory
WORKDIR /app

# Build checkpoint store
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
COPY sllm_store/requirements-build.txt /app/sllm_store/requirements-build.txt
RUN cd sllm_store && \
  pip install -r requirements-build.txt && \
  pip install setuptools wheel

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
RUN cd sllm_store && python3 setup.py bdist_wheel

# Build ServerlessLLM HTTP-based version
COPY requirements.txt requirements-worker.txt /app/
COPY pyproject.toml setup.py py.typed /app/
COPY sllm/serve /app/sllm/serve
COPY sllm/cli /app/sllm/cli
COPY README.md /app/
RUN python3 setup.py bdist_wheel

#################### RUNTIME IMAGE ####################
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set environment for HTTP-based architecture
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Default values for HTTP architecture
    STORAGE_PATH=/models \
    LOG_LEVEL=INFO \
    # Head node defaults
    HEAD_HOST=0.0.0.0 \
    HEAD_PORT=8080 \
    REDIS_HOST=redis \
    REDIS_PORT=6379 \
    # Worker node defaults  
    WORKER_HOST=0.0.0.0 \
    WORKER_PORT=8000

# Install additional runtime dependencies
RUN apt-get update -y && \
    apt-get install -y curl netcat-openbsd && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create conda environments optimized for HTTP architecture
RUN conda create -n head python=3.10 -y && \
    conda create -n worker python=3.10 -y

RUN conda run -n head pip install -U pip && \
    conda run -n worker pip install -U pip

# Copy and install updated requirements (no Ray dependencies)
COPY requirements.txt /app/
COPY requirements-worker.txt /app/

# Install head node dependencies (API gateway, Redis client)
RUN conda run -n head pip install -r /app/requirements.txt

# Install worker node dependencies (ML inference, HTTP client)
RUN conda run -n worker pip install -r /app/requirements-worker.txt

# Copy vLLM patch for worker (if needed)
COPY sllm_store/vllm_patch /app/vllm_patch

# Copy the built wheels from the builder
COPY --from=builder /app/sllm_store/dist /app/sllm_store/dist
COPY --from=builder /app/dist /app/dist

# Install ServerlessLLM packages in both environments
RUN conda run -n head pip install /app/sllm_store/dist/*.whl && \
    conda run -n head pip install /app/dist/*.whl

RUN conda run -n worker pip install /app/sllm_store/dist/*.whl && \
    conda run -n worker pip install /app/dist/*.whl

# Apply vLLM patch in worker environment
RUN conda run -n worker bash -c "cd /app && ./vllm_patch/patch.sh"

# Create model storage directory
RUN mkdir -p /models && \
    chmod 755 /models

# Copy the updated HTTP-based entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD /app/entrypoint.sh health

# Expose ports
# Head node: 8080 (API Gateway)
# Worker node: 8000 (Worker API)
EXPOSE 8080 8000

# Labels for container identification
LABEL org.opencontainers.image.title="ServerlessLLM HTTP" \
      org.opencontainers.image.description="HTTP-based distributed LLM serving platform" \
      org.opencontainers.image.version="2.0" \
      org.opencontainers.image.vendor="ServerlessLLM Team" \
      org.opencontainers.image.licenses="Apache-2.0"

# Set the entrypoint to our HTTP-based startup script
ENTRYPOINT ["/app/entrypoint.sh"]
