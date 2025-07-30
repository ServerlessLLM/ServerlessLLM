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

# Default values for HTTP-based architecture
DEFAULT_HEAD_HOST="0.0.0.0"
DEFAULT_HEAD_PORT="8343"  # Updated to match Python default
DEFAULT_REDIS_HOST="redis"
DEFAULT_REDIS_PORT="6379"
DEFAULT_WORKER_PORT="8001"  # Updated to match Python default
DEFAULT_STORAGE_PATH="/models"
DEFAULT_LOG_LEVEL="INFO"

# Source conda
source /opt/conda/etc/profile.d/conda.sh

# Function to initialize the head node
initialize_head_node() {
  # Activate head environment
  conda activate head

  # Set environment variables
  export REDIS_HOST="${REDIS_HOST:-$DEFAULT_REDIS_HOST}"
  export REDIS_PORT="${REDIS_PORT:-$DEFAULT_REDIS_PORT}"
  export LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
  HEAD_HOST="${HEAD_HOST:-$DEFAULT_HEAD_HOST}"
  HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"

  # Validate Redis connection
  timeout 30 bash -c "until echo > /dev/tcp/${REDIS_HOST}/${REDIS_PORT}; do sleep 1; done" || {
    echo "ERROR: Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
    exit 1
  }

  # Display and execute the command
  echo "Executing: $CMD"
  eval "$CMD"

  # Start sllm with any additional arguments passed to the script
  echo "Starting sllm with arguments: $@"
  exec sllm start head "$@"
}

# Function to initialize the worker node
initialize_worker_node() {
  # Parse sllm-store specific arguments
  STORE_ARGS=()
  WORKER_ARGS=()

  while [[ $# -gt 0 ]]; do
    case $1 in
      --mem-pool-size|--registration-required)
        STORE_ARGS+=("$1" "$2")
        shift 2
        ;;
      *)
        WORKER_ARGS+=("$1")
        shift
        ;;
    esac
  done

  # Activate worker environment
  conda activate worker

  # Set environment variables
  export STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  export LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"

  WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
  WORKER_PORT="${WORKER_PORT:-$DEFAULT_WORKER_PORT}"
  HEAD_NODE_URL="${HEAD_NODE_URL:-http://sllm_head:${DEFAULT_HEAD_PORT}}"

  # Validate required environment variables
  if [ -z "$HEAD_NODE_URL" ]; then
    echo "ERROR: HEAD_NODE_URL must be set to the head node's URL (e.g., http://sllm_head:8343)"
    exit 1
  fi

  # Create storage directory if it doesn't exist
  mkdir -p "$STORAGE_PATH"

  # Validate head node connection
  timeout 30 bash -c "
    while ! curl -s -o /dev/null -w '%{http_code}' ${HEAD_NODE_URL}/health | grep -q '200'; do
      sleep 2
    done
  " || {
    echo "WARNING: Cannot connect to head node at ${HEAD_NODE_URL}"
  }

  # Start worker with HTTP heartbeat to head node in background
  sllm start worker "${WORKER_ARGS[@]}" &

  # Start sllm-store with sllm-store specific arguments
  exec sllm-store start --storage-path="$STORAGE_PATH" "${STORE_ARGS[@]}"
}

# Health check function
health_check() {
  if [ "$MODE" == "HEAD" ]; then
    HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"
    curl -f "http://localhost:${HEAD_PORT}/health" || exit 1
  elif [ "$MODE" == "WORKER" ]; then
    WORKER_PORT="${WORKER_PORT:-$DEFAULT_WORKER_PORT}"
    curl -f "http://localhost:${WORKER_PORT}/health" || exit 1
  else
    echo "Unknown MODE for health check: $MODE"
    exit 1
  fi
}

# Print usage information
usage() {
  echo "ServerlessLLM HTTP-based Container Entrypoint"
  echo ""
  echo "Environment Variables:"
  echo "  MODE                 - Required: 'HEAD' or 'WORKER'"
  echo ""
  echo "Head Node Variables:"
  echo "  HEAD_HOST           - Host to bind to (default: 0.0.0.0)"
  echo "  HEAD_PORT           - Port to bind to (default: 8343)"
  echo "  REDIS_HOST          - Redis hostname (default: localhost)"
  echo "  REDIS_PORT          - Redis port (default: 6379)"
  echo ""
  echo "Worker Node Variables:"
  echo "  WORKER_HOST         - Host to bind to (default: 0.0.0.0)"
  echo "  WORKER_PORT         - Port to bind to (default: 8001)"
  echo "  HEAD_NODE_URL       - Head node URL (required, e.g., http://sllm_head:8343)"
  echo "  NODE_ID             - Unique worker node ID (default: hostname or generated)"
  echo ""
  echo "Common Variables:"
  echo "  STORAGE_PATH        - Model storage path (default: /models)"
  echo "  LOG_LEVEL           - Logging level (default: INFO)"
  echo ""
  echo "Special Commands:"
  echo "  health              - Run health check"
  echo "  help                - Show this help message"
}

# Handle special commands
case "$1" in
  "health")
    health_check
    exit 0
    ;;
  "help")
    usage
    exit 0
    ;;
esac

# Determine the node type and call the appropriate initialization function
if [ "$MODE" == "HEAD" ]; then
  initialize_head_node "$@"
elif [ "$MODE" == "WORKER" ]; then
  initialize_worker_node "$@"
else
  echo "ERROR: MODE must be set to either 'HEAD' or 'WORKER'"
  echo ""
  usage
  exit 1
fi
