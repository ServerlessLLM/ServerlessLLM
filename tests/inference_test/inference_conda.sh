#!/bin/bash

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda3/bin/activate
conda activate sllm

# Create models directory
echo "Creating models directory..."
mkdir -p models
echo "Models directory created at $(pwd)/models"

# Set environment variables
echo "Setting up environment variables..."
export MODEL_FOLDER=$(pwd)/models
export LLM_SERVER_URL=http://127.0.0.1:8343/
echo "MODEL_FOLDER is set to: $MODEL_FOLDER"
echo "LLM_SERVER_URL is set to: $LLM_SERVER_URL"
