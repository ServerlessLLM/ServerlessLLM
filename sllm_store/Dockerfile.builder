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
FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu20.04

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages for wget and HTTPS
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the Miniconda version to install
ENV MINICONDA_VERSION 24.3.0-0
ENV MINICONDA_SHA256 def595b1b182749df0974cddb5c8befe70664ace16403d7a7bf54467be5ea48b

# Download the Miniconda installer
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh

# Validate the installer with md5sum
RUN echo "${MINICONDA_SHA256} /tmp/miniconda.sh" | sha256sum -c -

# Install Miniconda
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda

# Clean up the installer script
RUN rm /tmp/miniconda.sh

# Initialize Conda for all shell types
RUN /opt/conda/bin/conda init

# Add /opt/conda/bin to PATH
ENV PATH /opt/conda/bin:$PATH

COPY requirements-build.txt .

# Optional: Install any other dependencies or setup the environment
RUN conda create -n py310 python=3.10
RUN /opt/conda/envs/py310/bin/pip install -r requirements-build.txt

# Add your library files
WORKDIR /app
COPY cmake ./cmake
COPY CMakeLists.txt .
COPY csrc ./csrc
COPY sllm_store ./sllm_store
COPY setup.py .
COPY pyproject.toml .
COPY MANIFEST.in .
COPY requirements.txt .
COPY README.md .
COPY proto ./proto

# ENTRYPOINT [ "/bin/bash", "-c" ]