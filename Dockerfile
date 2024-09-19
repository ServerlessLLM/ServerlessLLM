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
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for wget and HTTPS
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates git

# Set the working directory
WORKDIR /app

RUN conda install python=3.10
RUN pip install -U pip

# Install checkpoint store
# Option 1: Install from test.pypi.org
# RUN pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ serverless_llm_store==0.0.1.dev4
# Option 2: Install from local wheel. Please make sure to build the wheel first.
# COPY serverless_llm/store/dist/serverless_llm_store-0.0.1.dev4-cp310-cp310-linux_x86_64.whl /app/
# RUN pip install serverless_llm_store-0.0.1.dev4-cp310-cp310-linux_x86_64.whl
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
COPY serverless_llm/store/cmake /app/serverless_llm/store/cmake
COPY serverless_llm/store/CMakeLists.txt /app/serverless_llm/store/CMakeLists.txt
COPY serverless_llm/store/csrc /app/serverless_llm/store/csrc
COPY serverless_llm/store/serverless_llm_store /app/serverless_llm/store/serverless_llm_store
COPY serverless_llm/store/setup.py /app/serverless_llm/store/setup.py
COPY serverless_llm/store/pyproject.toml /app/serverless_llm/store/pyproject.toml
COPY serverless_llm/store/MANIFEST.in /app/serverless_llm/store/MANIFEST.in
COPY serverless_llm/store/setup.cfg /app/serverless_llm/store/setup.cfg
COPY serverless_llm/store/requirements.txt /app/serverless_llm/store/requirements.txt
COPY serverless_llm/store/README.md /app/serverless_llm/store/README.md
COPY serverless_llm/store/proto /app/serverless_llm/store/proto
RUN cd serverless_llm/store && pip install .

COPY requirements.txt /app/
COPY requirements-worker.txt /app/
RUN pip install -r requirements.txt

COPY pyproject.toml setup.py setup.cfg py.typed /app/
COPY serverless_llm/serve /app/serverless_llm/serve
COPY serverless_llm/cli /app/serverless_llm/cli
COPY examples examples
COPY README.md /app/
RUN pip install .

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV NODE_TYPE=HEAD
# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
