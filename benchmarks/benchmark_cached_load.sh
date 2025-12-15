#!/bin/bash

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

set -e

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 model_name models_directory num_repeats"
    echo "Example: $0 facebook/opt-6.7b ./models 30"
    echo ""
    echo "This script benchmarks cached model loading, where the same model"
    echo "is loaded repeatedly to measure cache/storage performance."
    echo "Unlike random load, only 1 model copy is saved per format."
    exit 1
fi

# Assign arguments to variables
MODEL_NAME=$1
MODELS_DIR=$2
NUM_REPEATS=$3

# Validate number of repeats
if [[ ! "$NUM_REPEATS" =~ ^[0-9]+$ ]]; then
    echo "Error: Number of repeats must be a positive integer."
    exit 1
fi

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory $MODELS_DIR does not exist. Please create it before running this script."
    exit 1
fi

# Function to download model (only 1 replica for cached test)
download_model() {
    local model_name="$1"
    local save_format="$2"
    local save_dir="$3"

    echo "Downloading $model_name model (1 copy for cached test)..."
    python3 download_models.py \
        --model-name "$model_name" \
        --save-format "$save_format" \
        --save-dir "$save_dir" \
        --num-replicas 1
}

# Function to run benchmark
run_benchmark() {
    local model_name="$1"
    local model_format="$2"
    local model_dir="$3"
    local num_repeats="$4"
    local benchmark_type="$5"

    echo "Running cached benchmark for $model_name ($num_repeats repeats)..."
    python3 test_loading.py \
        --model-name "$model_name" \
        --model-format "$model_format" \
        --model-dir "$model_dir" \
        --num-replicas "$num_repeats" \
        --benchmark-type "$benchmark_type"
}

# Function to clean models directory
clean_models_dir() {
    echo "Cleaning models directory..."
    rm -rf "$MODELS_DIR"/*
}

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Models directory does not exist. Please create it before running this script."
    exit 1
fi

# Clean models directory before starting benchmarks
clean_models_dir

# Start measuring performance for each format
for MODEL_FORMAT in safetensors sllm; do
    echo "Measuring $MODEL_FORMAT performance for $MODEL_NAME (cached test)..."
    download_model "$MODEL_NAME" "$MODEL_FORMAT" "$MODELS_DIR"
    run_benchmark "$MODEL_NAME" "$MODEL_FORMAT" "$MODELS_DIR" "$NUM_REPEATS" "cached"
    clean_models_dir
done

echo "Cached benchmark completed successfully."



