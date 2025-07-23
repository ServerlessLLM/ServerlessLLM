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
DEFAULT_REDIS_HOST="localhost"  # Updated to match Python default
DEFAULT_REDIS_PORT="6379"
DEFAULT_WORKER_PORT="8001"  # Updated to match Python default
DEFAULT_STORAGE_PATH="/models"
DEFAULT_LOG_LEVEL="INFO"

# Source conda
source /opt/conda/etc/profile.d/conda.sh

# Function to initialize the head node
initialize_head_node() {
  echo "Initializing HTTP-based head node..."

  # Activate head environment
  echo "Activating head conda environment..."
  conda activate head

  # Set environment variables
  export REDIS_HOST="${REDIS_HOST:-$DEFAULT_REDIS_HOST}"
  export REDIS_PORT="${REDIS_PORT:-$DEFAULT_REDIS_PORT}"
  export STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  export LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"

  HEAD_HOST="${HEAD_HOST:-$DEFAULT_HEAD_HOST}"
  HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"

  # Validate Redis connection
  echo "Validating Redis connection to ${REDIS_HOST}:${REDIS_PORT}..."
  timeout 30 bash -c "until echo > /dev/tcp/${REDIS_HOST}/${REDIS_PORT}; do sleep 1; done" || {
    echo "ERROR: Cannot connect to Redis at ${REDIS_HOST}:${REDIS_PORT}"
    echo "Please ensure Redis is running and accessible"
    exit 1
  }
  echo "Redis connection validated successfully"

  # Create storage directory if it doesn't exist
  mkdir -p "$STORAGE_PATH"
  echo "Storage path: $STORAGE_PATH"

  # Start sllm-serve head node with HTTP API gateway
  echo "Starting ServerlessLLM head node on ${HEAD_HOST}:${HEAD_PORT}"
  echo "Redis: ${REDIS_HOST}:${REDIS_PORT}"
  echo "Log level: ${LOG_LEVEL}"

  exec sllm-serve head \
    --host="$HEAD_HOST" \
    --port="$HEAD_PORT" \
    --redis-host="$REDIS_HOST" \
    --redis-port="$REDIS_PORT" \
    "$@"
}

# Function to initialize the worker node
initialize_worker_node() {
  echo "Initializing HTTP-based worker node..."

  # Activate worker environment
  echo "Activating worker conda environment..."
  conda activate worker

  # Set environment variables
  export STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  export LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"

  WORKER_HOST="${WORKER_HOST:-0.0.0.0}"
  WORKER_PORT="${WORKER_PORT:-$DEFAULT_WORKER_PORT}"
  HEAD_NODE_URL="${HEAD_NODE_URL:-http://sllm_head:${DEFAULT_HEAD_PORT}}"

  # Get node ID from environment or generate one
  if [ -z "$NODE_ID" ]; then
    NODE_ID="${HOSTNAME:-worker-$(date +%s)}"
    echo "Generated NODE_ID: $NODE_ID"
  fi

  # Validate required environment variables
  if [ -z "$HEAD_NODE_URL" ]; then
    echo "ERROR: HEAD_NODE_URL must be set to the head node's URL (e.g., http://sllm_head:8343)"
    exit 1
  fi

  # Create storage directory if it doesn't exist
  mkdir -p "$STORAGE_PATH"
  echo "Storage path: $STORAGE_PATH"

  # Validate head node connection
  echo "Validating head node connection to ${HEAD_NODE_URL}..."
  timeout 30 bash -c "
    while ! curl -s -o /dev/null -w '%{http_code}' ${HEAD_NODE_URL}/health | grep -q '200'; do
      echo 'Waiting for head node to be ready...'
      sleep 2
    done
  " || {
    echo "WARNING: Cannot connect to head node at ${HEAD_NODE_URL}"
    echo "Worker will start anyway and attempt to connect during runtime"
  }
  echo "Head node connection validated successfully"

  # Start worker with HTTP heartbeat to head node in background
  echo "Starting ServerlessLLM worker node on ${WORKER_HOST}:${WORKER_PORT}"
  echo "Head node: ${HEAD_NODE_URL}"
  echo "Node ID: ${NODE_ID}"
  echo "Storage: ${STORAGE_PATH}"
  echo "Log level: ${LOG_LEVEL}"

  sllm-serve worker \
    --host="$WORKER_HOST" \
    --port="$WORKER_PORT" \
    --node-id="$NODE_ID" \
    --head-node-url="$HEAD_NODE_URL" &

  # Start sllm-store with any additional arguments passed to the script
  echo "Starting sllm-store with arguments: --storage-path=$STORAGE_PATH $@"
  exec sllm-store start --storage-path="$STORAGE_PATH" "$@"
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
