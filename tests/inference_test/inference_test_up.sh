#!/bin/bash

SLLM_ROOT=$(git rev-parse --show-toplevel)

# Navigate to docker directory and start containers
cd $SLLM_ROOT/examples/docker
docker compose up -d --build

# Go back home
cd

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate sllm

# Set environment variables
export MODEL_FOLDER=/home/ryan/models
export LLM_SERVER_URL=http://127.0.0.1:8343/
