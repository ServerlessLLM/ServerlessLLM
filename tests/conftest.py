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
            labels={"deployment_id": "test-model:vllm", "type": "inference"},
        )
    )
    client.cancel_instance = AsyncMock()
    client.get_instance = AsyncMock(
        return_value=InstanceInfo(
            instance_id="test-instance-1",
            name="test-model-abc12345",
            status="RUNNING",
            endpoint="192.168.1.10:8080",
            labels={"deployment_id": "test-model:vllm", "type": "inference"},
        )
    )
    client.get_deployment_instances = AsyncMock(return_value=[])
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
# Router Fixtures
# ============================================================================ #


@pytest.fixture
def mock_autoscaler():
    """Create a mock Autoscaler for router tests."""
    autoscaler = MagicMock()
    autoscaler.receive_metrics = MagicMock()
    return autoscaler


@pytest.fixture
def mock_http_client():
    """Create a mock HTTP client for router tests."""
    client = MagicMock()

    # Create mock response context manager
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={})

    # Setup the async context manager
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    client.post = MagicMock(return_value=mock_cm)
    client.close = AsyncMock()

    return client


@pytest.fixture
def router(database, mock_autoscaler, mock_http_client):
    """Create a Router instance for testing."""
    from sllm.router import Router, RouterConfig

    config = RouterConfig(
        max_buffer_size=10,
        cold_start_timeout=30.0,
        request_timeout=60.0,
    )
    router = Router(
        database=database,
        autoscaler=mock_autoscaler,
        config=config,
    )
    # Inject mock HTTP client
    router._session = mock_http_client
    return router


@pytest.fixture
def router_small_buffer(database, mock_autoscaler, mock_http_client):
    """Create a Router with small buffer for testing buffer full scenarios."""
    from sllm.router import Router, RouterConfig

    config = RouterConfig(
        max_buffer_size=2,  # Small buffer
        cold_start_timeout=30.0,
        request_timeout=60.0,
    )
    router = Router(
        database=database,
        autoscaler=mock_autoscaler,
        config=config,
    )
    router._session = mock_http_client
    return router


@pytest.fixture
def router_short_timeout(database, mock_autoscaler, mock_http_client):
    """Create a Router with short timeout for testing timeout scenarios."""
    from sllm.router import Router, RouterConfig

    config = RouterConfig(
        max_buffer_size=10,
        cold_start_timeout=0.1,  # Very short timeout
        request_timeout=0.1,
    )
    router = Router(
        database=database,
        autoscaler=mock_autoscaler,
        config=config,
    )
    router._session = mock_http_client
    return router


# ============================================================================ #
# Autoscaler Fixtures
# ============================================================================ #


@pytest.fixture
def autoscaler_with_metrics(database):
    """Create an Autoscaler with receive_metrics support for testing."""
    from sllm.autoscaler import AutoScaler

    autoscaler = AutoScaler(database=database)

    # Add helper methods for tests
    def get_metrics(deployment_id: str):
        metrics = autoscaler._metrics.get(deployment_id)
        if metrics:
            return {
                "buffer_len": metrics.buffer_len,
                "in_flight": metrics.in_flight,
            }
        return {"buffer_len": 0, "in_flight": 0}

    def get_total_demand(deployment_id: str):
        metrics = autoscaler._metrics.get(deployment_id)
        if metrics:
            return metrics.total_demand
        return 0

    autoscaler.get_metrics = get_metrics
    autoscaler.get_total_demand = get_total_demand

    return autoscaler


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
# Sample Deployment Data
# ============================================================================ #


@pytest.fixture
def sample_deployment_data():
    """Sample deployment registration data."""
    return {
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
def sample_deployment(database, sample_deployment_data):
    """Create a sample deployment in the database."""
    deployment = database.create_deployment(**sample_deployment_data)
    return deployment


# Aliases for backward compatibility with existing tests
@pytest.fixture
def sample_model_data(sample_deployment_data):
    """Alias for sample_deployment_data."""
    return sample_deployment_data


@pytest.fixture
def sample_model(sample_deployment):
    """Alias for sample_deployment."""
    return sample_deployment
