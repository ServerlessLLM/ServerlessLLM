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
"""
Test fixtures for ServerlessLLM v1-beta.
"""

import asyncio
import os
import sys
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the project root directory to the Python path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


# ============================================================================ #
# Event Loop Fixture
# ============================================================================ #


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================ #
# Database Fixtures
# ============================================================================ #


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    # Also clean up WAL and SHM files
    for suffix in ["-wal", "-shm"]:
        wal_path = db_path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def database(temp_db_path):
    """Create a Database instance with a temporary file."""
    from sllm.database import Database

    db = Database(temp_db_path)
    yield db
    db.close()


# ============================================================================ #
# Mock Pylet Client Fixtures
# ============================================================================ #


@pytest.fixture
def mock_pylet_client():
    """Create a mock PyletClient for unit tests."""
    from sllm.pylet_client import InstanceInfo, WorkerInfo

    client = MagicMock()
    client.connect = AsyncMock()
    client.close = AsyncMock()
    client.is_connected = True

    # Default return values
    client.submit = AsyncMock(
        return_value=InstanceInfo(
            instance_id="test-instance-1",
            name="test-model-abc12345",
            status="PENDING",
            endpoint=None,
            labels={"model_id": "test-model:vllm", "type": "inference"},
        )
    )
    client.cancel_instance = AsyncMock()
    client.get_instance = AsyncMock(
        return_value=InstanceInfo(
            instance_id="test-instance-1",
            name="test-model-abc12345",
            status="RUNNING",
            endpoint="192.168.1.10:8080",
            labels={"model_id": "test-model:vllm", "type": "inference"},
        )
    )
    client.get_model_instances = AsyncMock(return_value=[])
    client.get_store_instances = AsyncMock(return_value=[])
    client.list_workers = AsyncMock(
        return_value=[
            WorkerInfo(
                name="worker-0",
                status="READY",
                total_gpus=4,
                available_gpus=[0, 1, 2, 3],
                labels={},
            )
        ]
    )

    return client


# ============================================================================ #
# LoadBalancer Fixtures
# ============================================================================ #


@pytest.fixture
def load_balancer():
    """Create a LoadBalancer instance for testing."""
    from sllm.load_balancer import LBConfig, LoadBalancer

    config = LBConfig(
        max_buffer_size=10,
        cold_start_timeout=30.0,
        request_timeout=60.0,
    )
    lb = LoadBalancer(model_id="test-model:vllm", config=config)
    return lb


@pytest.fixture
def lb_registry():
    """Create a LoadBalancerRegistry instance."""
    from sllm.lb_registry import LoadBalancerRegistry

    registry = LoadBalancerRegistry()
    yield registry
    # Cleanup
    asyncio.get_event_loop().run_until_complete(registry.shutdown())


# ============================================================================ #
# StorageManager Fixtures
# ============================================================================ #


@pytest.fixture
def storage_manager(database, mock_pylet_client):
    """Create a StorageManager instance."""
    from sllm.storage_manager import StorageManager

    manager = StorageManager(
        database=database,
        pylet_client=mock_pylet_client,
        storage_path="/models",
    )
    return manager


# ============================================================================ #
# Sample Model Data
# ============================================================================ #


@pytest.fixture
def sample_model_data():
    """Sample model registration data."""
    return {
        "model_id": "facebook/opt-125m:vllm",
        "model_name": "facebook/opt-125m",
        "backend": "vllm",
        "backend_config": {
            "tensor_parallel_size": 1,
            "max_model_len": 2048,
        },
        "min_replicas": 0,
        "max_replicas": 4,
        "target_pending_requests": 10,
        "keep_alive_seconds": 60,
    }


@pytest.fixture
def sample_model(database, sample_model_data):
    """Create a sample model in the database."""
    model = database.create_model(**sample_model_data)
    return model
