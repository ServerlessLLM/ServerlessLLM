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

# Default values
DEFAULT_RAY_PORT=6379
DEFAULT_RAY_RESOURCES_HEAD='{"control_node": 1}'
DEFAULT_RAY_NUM_CPUS=20
DEFAULT_RAY_HEAD_ADDRESS="ray_head:6379"
DEFAULT_STORAGE_PATH="/models"

# Function to initialize the head node
initialize_head_node() {
  echo "Initializing head node..."
  RAY_PORT="${RAY_PORT:-$DEFAULT_RAY_PORT}"
  RAY_RESOURCES="${RAY_RESOURCES:-$DEFAULT_RAY_RESOURCES_HEAD}"
  RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"
  
  # Construct the command
  CMD="ray start --head --port=$RAY_PORT --resources='$RAY_RESOURCES' --num-cpus=$RAY_NUM_CPUS --block"
  
  # Display and execute the command
  echo "Executing: $CMD"
  eval "$CMD"
}

# Function to initialize the worker node
initialize_worker_node() {
  echo "Initializing worker node..."

  # Patch the vLLM code
  VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(os.path.abspath(vllm.__file__)))")
  patch -p2 -d $VLLM_PATH < ./serverless_llm/store/vllm_patch/sllm_load.patch

  # Start checkpoint store
  STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  # TODO: Temporary remove the registration required flag, as registration is not working for vLLM backend
  # CMD="sllm-store-server -storage_path=$STORAGE_PATH -registration_required=true &"
  CMD="sllm-store-server -storage_path=$STORAGE_PATH &"
  echo "Executing: $CMD"
  eval "$CMD"

  # Start the worker
  RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-$DEFAULT_RAY_HEAD_ADDRESS}"

  if [ -z "$WORKER_ID" ]; then
    echo "WORKER_ID must be set"
    exit 1
  fi

  RAY_RESOURCES='{"worker_node": 1, "worker_id_'$WORKER_ID'": 1}'
  
  # Construct the command
  CMD="ray start --address=$RAY_HEAD_ADDRESS --resources='$RAY_RESOURCES' --block"
  
  # Display and execute the command
  echo "Executing: $CMD"
  eval "$CMD"
}

# Determine the node type and call the appropriate initialization function
if [ "$NODE_TYPE" == "HEAD" ]; then
  initialize_head_node
elif [ "$NODE_TYPE" == "WORKER" ]; then
  initialize_worker_node
else
  echo "NODE_TYPE must be set to either HEAD or WORKER"
  exit 1
fi
