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
DEFAULT_HEAD_HOST="0.0.0.0"
DEFAULT_HEAD_PORT="8343"
DEFAULT_STORAGE_PATH="/models"
DEFAULT_DATABASE_PATH="/var/lib/sllm/state.db"
DEFAULT_PYLET_ENDPOINT="http://pylet_head:8000"
DEFAULT_PYLET_HEAD="pylet_head:8000"
DEFAULT_GPU_UNITS="1"
DEFAULT_LOG_LEVEL="INFO"

# HEAD mode: Run SLLM control plane (v1-beta)
initialize_head_node() {
  source /opt/venvs/head/bin/activate

  HEAD_HOST="${HEAD_HOST:-$DEFAULT_HEAD_HOST}"
  HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"
  STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  DATABASE_PATH="${SLLM_DATABASE_PATH:-$DEFAULT_DATABASE_PATH}"
  PYLET_ENDPOINT="${PYLET_ENDPOINT:-$DEFAULT_PYLET_ENDPOINT}"
  LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"

  export LOG_LEVEL

  # Create database directory
  mkdir -p "$(dirname "$DATABASE_PATH")"

  # Wait for Pylet head to be available
  echo "Waiting for Pylet head at ${PYLET_ENDPOINT}..."
  timeout 60 bash -c "
    until curl -s -o /dev/null -w '%{http_code}' ${PYLET_ENDPOINT}/workers 2>/dev/null | grep -q '200'; do
      sleep 2
    done
  " || echo "WARNING: Could not connect to Pylet head at ${PYLET_ENDPOINT}"

  echo "Starting SLLM head (v1-beta)..."
  echo "  Host: ${HEAD_HOST}"
  echo "  Port: ${HEAD_PORT}"
  echo "  Pylet: ${PYLET_ENDPOINT}"
  echo "  Database: ${DATABASE_PATH}"
  echo "  Storage: ${STORAGE_PATH}"

  exec sllm start \
    --host "$HEAD_HOST" \
    --port "$HEAD_PORT" \
    --pylet-endpoint "$PYLET_ENDPOINT" \
    --database-path "$DATABASE_PATH" \
    --storage-path "$STORAGE_PATH" \
    "$@"
}

# PYLET_WORKER mode: Run Pylet worker (for GPU nodes)
initialize_pylet_worker() {
  # Use minimal pylet venv (pylet spawns processes in separate venvs)
  source /opt/venvs/pylet/bin/activate

  PYLET_HEAD="${PYLET_HEAD:-$DEFAULT_PYLET_HEAD}"
  GPU_UNITS="${GPU_UNITS:-$DEFAULT_GPU_UNITS}"
  STORAGE_PATH="${STORAGE_PATH:-$DEFAULT_STORAGE_PATH}"
  LOG_LEVEL="${LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"

  export LOG_LEVEL

  # Create storage directory
  mkdir -p "$STORAGE_PATH"

  # Wait for Pylet head to be available
  echo "Waiting for Pylet head at http://${PYLET_HEAD}..."
  timeout 60 bash -c "
    until curl -s -o /dev/null -w '%{http_code}' http://${PYLET_HEAD}/workers 2>/dev/null | grep -q '200'; do
      sleep 2
    done
  " || {
    echo "ERROR: Could not connect to Pylet head at ${PYLET_HEAD}"
    exit 1
  }

  echo "Starting Pylet worker..."
  echo "  Pylet head: ${PYLET_HEAD}"
  echo "  GPU units: ${GPU_UNITS}"
  echo "  Storage: ${STORAGE_PATH}"

  exec pylet start --head "$PYLET_HEAD" --gpu-units "$GPU_UNITS"
}

# Health check function
health_check() {
  case "$MODE" in
    "HEAD")
      HEAD_PORT="${HEAD_PORT:-$DEFAULT_HEAD_PORT}"
      curl -f "http://localhost:${HEAD_PORT}/health" || exit 1
      ;;
    "PYLET_WORKER")
      # Pylet worker doesn't have HTTP health endpoint, check process
      pgrep -f "pylet" > /dev/null || exit 1
      ;;
    *)
      echo "Unknown MODE for health check: $MODE"
      exit 1
      ;;
  esac
}

# Print usage information
usage() {
  echo "ServerlessLLM v1-beta Container"
  echo ""
  echo "Modes:"
  echo "  MODE=HEAD          Run SLLM control plane"
  echo "  MODE=PYLET_WORKER  Run Pylet worker (GPU node)"
  echo ""
  echo "HEAD mode environment variables:"
  echo "  HEAD_HOST          Host to bind to (default: 0.0.0.0)"
  echo "  HEAD_PORT          Port to bind to (default: 8343)"
  echo "  PYLET_ENDPOINT     Pylet head URL (default: http://pylet_head:8000)"
  echo "  SLLM_DATABASE_PATH SQLite database path (default: /var/lib/sllm/state.db)"
  echo "  STORAGE_PATH       Model storage path (default: /models)"
  echo "  LOG_LEVEL          Logging level (default: INFO)"
  echo ""
  echo "PYLET_WORKER mode environment variables:"
  echo "  PYLET_HEAD         Pylet head address (default: pylet_head:8000)"
  echo "  GPU_UNITS          Number of GPUs to register (default: 1)"
  echo "  STORAGE_PATH       Model storage path (default: /models)"
  echo "  LOG_LEVEL          Logging level (default: INFO)"
  echo ""
  echo "Special commands:"
  echo "  health             Run health check"
  echo "  help               Show this help message"
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
case "$MODE" in
  "HEAD")
    initialize_head_node "$@"
    ;;
  "PYLET_WORKER")
    initialize_pylet_worker "$@"
    ;;
  *)
    echo "ERROR: MODE must be set to 'HEAD' or 'PYLET_WORKER'"
    echo ""
    usage
    exit 1
    ;;
esac
