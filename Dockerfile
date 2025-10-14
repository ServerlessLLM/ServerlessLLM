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

# Adapted from https://github.com/vllm-project/vllm/blob/23c1b10a4c8cd77c5b13afa9242d67ffd055296b/Dockerfile
ARG CUDA_VERSION=12.1.1
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS builder
ARG CUDA_VERSION=12.1.1
ARG PYTHON_VERSION=3.10
ARG TARGETPLATFORM
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git sudo build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/mf.sh \
    && bash /tmp/mf.sh -b -p ${CONDA_DIR} \
    && rm /tmp/mf.sh
ENV PATH=${CONDA_DIR}/bin:$PATH
SHELL ["/bin/bash", "-lc"]
RUN conda create -y -n build python=3.10 pip setuptools wheel \
    && conda run -n build python -V \
    && conda run -n build pip -V
# Set the working directory
WORKDIR /app

# Build checkpoint store
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
COPY sllm_store/requirements-build.txt /app/sllm_store/requirements-build.txt
RUN cd sllm_store && \
  conda run -n build python -m pip install -r requirements-build.txt && \
  conda run -n build python -m pip install setuptools wheel

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
RUN cd sllm_store && conda run -n build python setup.py bdist_wheel

COPY requirements.txt requirements-worker.txt /app/
COPY pyproject.toml setup.py py.typed /app/
COPY sllm/backends /app/sllm/backends
COPY sllm/ft_backends /app/sllm/ft_backends
COPY sllm/cli /app/sllm/cli
COPY sllm/routers /app/sllm/routers
COPY sllm/schedulers /app/sllm/schedulers
COPY sllm/*.py /app/sllm/
COPY README.md /app/
RUN conda run -n build python setup.py bdist_wheel

# Stage 2: Runner with conda environments
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set the working directory
WORKDIR /app

# Create conda environments for head and worker
RUN conda create -n head python=3.10 -y && \
    conda create -n worker python=3.10 -y

RUN conda run -n head pip install -U pip
RUN conda run -n worker pip install -U pip

# Copy requirements files
COPY requirements.txt /app/

RUN conda run -n head pip install -r /app/requirements.txt
COPY requirements-worker.txt /app/

RUN conda run -n worker pip install -r /app/requirements-worker.txt

# Copy vllm patch for worker
COPY sllm_store/vllm_patch /app/vllm_patch

# Copy the built wheels from the builder
COPY --from=builder /app/sllm_store/dist /app/sllm_store/dist
COPY --from=builder /app/dist /app/dist

# Install packages in head environment
RUN conda run -n head pip install /app/sllm_store/dist/*.whl && \
    conda run -n head pip install /app/dist/*.whl

# Install packages in worker environment
RUN conda run -n worker pip install /app/sllm_store/dist/*.whl && \
    conda run -n worker pip install /app/dist/*.whl

# Apply vLLM patch in worker environment
RUN conda run -n worker bash -c "cd /app && ./vllm_patch/patch.sh"

# Copy the entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set the entrypoint directly to the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
