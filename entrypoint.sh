#!/bin/bash

set -e

# Default values
DEFAULT_RAY_PORT=6379
DEFAULT_RAY_RESOURCES_HEAD='{"control_node": 1}'
DEFAULT_RAY_NUM_CPUS=20
DEFAULT_RAY_HEAD_ADDRESS="sllm_head:6379"
DEFAULT_STORAGE_PATH="/models"

# Source conda
source /opt/conda/etc/profile.d/conda.sh

# Function to initialize the head node
initialize_head_node() {
  echo "Initializing head node..."

  # Activate head environment
  echo "Activating head conda environment..."
  conda activate head

  RAY_PORT="${RAY_PORT:-$DEFAULT_RAY_PORT}"
  RAY_RESOURCES="${RAY_RESOURCES:-$DEFAULT_RAY_RESOURCES_HEAD}"
  RAY_NUM_CPUS="${RAY_NUM_CPUS:-$DEFAULT_RAY_NUM_CPUS}"

  # Construct the command
  CMD="ray start --head --port=$RAY_PORT --resources='$RAY_RESOURCES' --num-cpus=$RAY_NUM_CPUS"

  # Add node IP if specified
  if [ ! -z "$RAY_NODE_IP" ]; then
    echo "Using specified node IP: $RAY_NODE_IP"
    CMD="$CMD --node-ip-address=$RAY_NODE_IP"
  else
    echo "No node IP specified. Ray will attempt to determine the best IP automatically."
  fi

  # Display and execute the command
  echo "Executing: $CMD"
  eval "$CMD"


echo "Starting SLLM API server..."


nohup python /opt/conda/envs/head/lib/python3.10/site-packages/sllm/serve/commands/serve/sllm_serve.py start > /tmp/sllm_api.log 2>&1 &


sleep 5


if pgrep -f "sllm_serve.py" > /dev/null; then
    echo " SLLM API server started successfully on port 8343"
else
    echo "Failed to start SLLM API server"
    echo "Check logs: /tmp/sllm_api.log"
fi

echo "SLLM head node is ready!"


tail -f /dev/null
}

# Function to initialize the worker node  
initialize_worker_node() {
  echo "Initializing worker node..."

  # Activate worker environment
  echo "Activating worker conda environment..."
  conda activate worker

  # Start the worker
  RAY_HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-$DEFAULT_RAY_HEAD_ADDRESS}"

  if [ -z "$WORKER_ID" ]; then
    echo "WORKER_ID must be set"
    exit 1
  fi

  RAY_RESOURCES='{"worker_node": 1, "worker_id_'$WORKER_ID'": 1}'

  # Construct the command
  CMD="ray start --address=$RAY_HEAD_ADDRESS --resources='$RAY_RESOURCES'"

  # Add node IP if specified
  if [ ! -z "$RAY_NODE_IP" ]; then
    echo "Using specified node IP: $RAY_NODE_IP"
    CMD="$CMD --node-ip-address=$RAY_NODE_IP"
  else
    echo "No node IP specified. Ray will attempt to determine the best IP automatically."
  fi

  # Display and execute the command
  echo "Executing: $CMD"
  eval "$CMD"

  
  STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  echo "Starting sllm-store with arguments: --storage-path=$STORAGE_PATH $@"
  exec sllm-store start --storage-path=$STORAGE_PATH "$@"
}

# Determine the node type and call the appropriate initialization function
if [ "$MODE" == "HEAD" ]; then
  initialize_head_node "$@"
elif [ "$MODE" == "WORKER" ]; then
  initialize_worker_node "$@"
else
  echo "MODE must be set to either HEAD or WORKER"
  exit 1
fi
