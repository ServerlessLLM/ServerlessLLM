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
"""Tests for the SQLite database layer."""

import json

import pytest

from sllm.database import Deployment


class TestDatabase:
    """Tests for Database class."""

    def test_create_deployment(self, database, sample_deployment_data):
        """Test deployment creation."""
        deployment = database.create_deployment(**sample_deployment_data)

        assert deployment.id == "facebook/opt-125m:vllm"
        assert deployment.model_name == "facebook/opt-125m"
        assert deployment.backend == "vllm"
        assert deployment.status == "active"
        assert deployment.desired_replicas == 0
        assert deployment.min_replicas == 0
        assert deployment.max_replicas == 4
        assert deployment.target_pending_requests == 10
        assert deployment.keep_alive_seconds == 60
        assert deployment.backend_config["tensor_parallel_size"] == 1

    def test_get_deployment(self, database, sample_deployment_data):
        """Test getting a deployment by model_name and backend."""
        created = database.create_deployment(**sample_deployment_data)
        retrieved = database.get_deployment(created.model_name, created.backend)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.model_name == created.model_name

    def test_get_deployment_by_id(self, database, sample_deployment_data):
        """Test getting a deployment by ID."""
        created = database.create_deployment(**sample_deployment_data)
        retrieved = database.get_deployment_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_deployment_not_found(self, database):
        """Test getting a non-existent deployment."""
        deployment = database.get_deployment("nonexistent", "vllm")
        assert deployment is None

    def test_get_all_deployments(self, database):
        """Test getting all deployments."""
        # Create multiple deployments
        database.create_deployment(
            model_name="model1",
            backend="vllm",
            min_replicas=0,
            max_replicas=2,
        )
        database.create_deployment(
            model_name="model2",
            backend="sglang",
            min_replicas=1,
            max_replicas=4,
        )

        deployments = database.get_all_deployments()
        assert len(deployments) == 2
        assert {d.id for d in deployments} == {"model1:vllm", "model2:sglang"}

    def test_get_active_deployments(self, database):
        """Test getting only active deployments."""
        # Create deployments with active status
        database.create_deployment(
            model_name="active1", backend="vllm", initial_status="active"
        )
        database.create_deployment(
            model_name="active2", backend="vllm", initial_status="active"
        )

        # Update one to deleting status
        conn = database._get_connection()
        conn.execute(
            "UPDATE deployments SET status = ? WHERE id = ?",
            ("deleting", "active2:vllm"),
        )

        active_deployments = database.get_active_deployments()
        assert len(active_deployments) == 1
        assert active_deployments[0].id == "active1:vllm"

    def test_update_desired_replicas(self, database, sample_deployment):
        """Test updating desired replicas."""
        database.update_desired_replicas(sample_deployment.id, 3)

        updated = database.get_deployment_by_id(sample_deployment.id)
        assert updated.desired_replicas == 3

    def test_update_deployment_status(self, database, sample_deployment):
        """Test updating deployment status."""
        database.update_deployment_status(sample_deployment.id, "deleting")

        updated = database.get_deployment_by_id(sample_deployment.id)
        assert updated.status == "deleting"

    def test_delete_deployment(self, database, sample_deployment):
        """Test deleting a deployment."""
        database.delete_deployment(sample_deployment.id)

        deleted = database.get_deployment_by_id(sample_deployment.id)
        assert deleted is None

    def test_duplicate_deployment_raises_error(
        self, database, sample_deployment_data
    ):
        """Test that creating a duplicate deployment raises an error."""
        database.create_deployment(**sample_deployment_data)

        with pytest.raises(ValueError):
            database.create_deployment(**sample_deployment_data)

    def test_deployment_make_id(self):
        """Test Deployment.make_id static method."""
        deployment_id = Deployment.make_id("meta-llama/Llama-3.1-8B", "vllm")
        assert deployment_id == "meta-llama/Llama-3.1-8B:vllm"


class TestNodeStorage:
    """Tests for node storage operations."""

    def test_upsert_node_storage(self, database):
        """Test upserting node storage info."""
        database.upsert_node_storage(
            node_name="worker-0",
            sllm_store_endpoint="192.168.1.10:8073",
            cached_models=["model1", "model2"],
        )

        node = database.get_node_storage("worker-0")
        assert node is not None
        assert node.sllm_store_endpoint == "192.168.1.10:8073"
        assert node.cached_models == ["model1", "model2"]

    def test_upsert_node_storage_update(self, database):
        """Test updating existing node storage info."""
        database.upsert_node_storage(
            node_name="worker-0",
            sllm_store_endpoint="192.168.1.10:8073",
            cached_models=["model1"],
        )

        # Update with new info
        database.upsert_node_storage(
            node_name="worker-0",
            sllm_store_endpoint="192.168.1.10:8073",
            cached_models=["model1", "model2", "model3"],
        )

        node = database.get_node_storage("worker-0")
        assert len(node.cached_models) == 3

    def test_get_all_nodes(self, database):
        """Test getting all node storage info."""
        database.upsert_node_storage(
            node_name="worker-0",
            sllm_store_endpoint="192.168.1.10:8073",
            cached_models=[],
        )
        database.upsert_node_storage(
            node_name="worker-1",
            sllm_store_endpoint="192.168.1.11:8073",
            cached_models=[],
        )

        nodes = database.get_all_node_storage()
        assert len(nodes) == 2
        assert {n.node_name for n in nodes} == {"worker-0", "worker-1"}

    def test_get_nodes_with_model(self, database):
        """Test getting nodes that have a specific model cached."""
        database.upsert_node_storage(
            node_name="worker-0",
            sllm_store_endpoint="192.168.1.10:8073",
            cached_models=["model1", "model2"],
        )
        database.upsert_node_storage(
            node_name="worker-1",
            sllm_store_endpoint="192.168.1.11:8073",
            cached_models=["model2", "model3"],
        )
        database.upsert_node_storage(
            node_name="worker-2",
            sllm_store_endpoint="192.168.1.12:8073",
            cached_models=["model3"],
        )

        nodes_with_model2 = database.get_nodes_with_model("model2")
        assert set(nodes_with_model2) == {"worker-0", "worker-1"}

        nodes_with_model1 = database.get_nodes_with_model("model1")
        assert nodes_with_model1 == ["worker-0"]


class TestDeploymentEndpoints:
    """Tests for deployment_endpoints table operations."""

    def test_add_deployment_endpoint(self, database):
        """Test adding an endpoint for a deployment."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")

        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert endpoints == ["192.168.1.10:8080"]

    def test_add_multiple_endpoints(self, database):
        """Test adding multiple endpoints for a deployment."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.11:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.12:8080")

        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert len(endpoints) == 3
        assert set(endpoints) == {
            "192.168.1.10:8080",
            "192.168.1.11:8080",
            "192.168.1.12:8080",
        }

    def test_add_duplicate_endpoint_updates(self, database):
        """Test that adding duplicate endpoint updates instead of creating."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")

        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert len(endpoints) == 1

    def test_remove_deployment_endpoint(self, database):
        """Test removing an endpoint."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.11:8080")

        database.remove_deployment_endpoint(
            "test-model:vllm", "192.168.1.10:8080"
        )

        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert endpoints == ["192.168.1.11:8080"]

    def test_remove_nonexistent_endpoint(self, database):
        """Test that removing non-existent endpoint doesn't raise."""
        # Should not raise
        database.remove_deployment_endpoint(
            "test-model:vllm", "192.168.1.99:8080"
        )

    def test_mark_endpoint_unhealthy(self, database):
        """Test marking endpoint as unhealthy."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.11:8080")

        database.mark_endpoint_unhealthy("test-model:vllm", "192.168.1.10:8080")

        # Only healthy endpoints should be returned
        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert endpoints == ["192.168.1.11:8080"]

    def test_get_deployment_endpoints_empty(self, database):
        """Test getting endpoints for deployment with no endpoints."""
        endpoints = database.get_deployment_endpoints("unknown-model:vllm")
        assert endpoints == []

    def test_get_deployment_endpoints_only_healthy(self, database):
        """Test that only healthy endpoints are returned."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.11:8080")
        database.mark_endpoint_unhealthy("test-model:vllm", "192.168.1.11:8080")

        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert len(endpoints) == 1
        assert "192.168.1.10:8080" in endpoints

    def test_delete_deployment_endpoints(self, database):
        """Test deleting all endpoints for a deployment."""
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.11:8080")

        database.delete_deployment_endpoints("test-model:vllm")

        endpoints = database.get_deployment_endpoints("test-model:vllm")
        assert endpoints == []

    def test_endpoints_isolated_per_deployment(self, database):
        """Test that endpoints are isolated per deployment."""
        database.add_deployment_endpoint("model-a:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("model-b:vllm", "192.168.1.20:8080")

        endpoints_a = database.get_deployment_endpoints("model-a:vllm")
        endpoints_b = database.get_deployment_endpoints("model-b:vllm")

        assert endpoints_a == ["192.168.1.10:8080"]
        assert endpoints_b == ["192.168.1.20:8080"]

    def test_endpoints_returned_in_sorted_order(self, database):
        """Test that endpoints are returned in consistent sorted order.

        This is important for round-robin load balancing - without consistent
        ordering, the same round-robin index could select different endpoints
        on different calls.
        """
        # Add endpoints in reverse/random order
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.30:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.20:8080")

        endpoints = database.get_deployment_endpoints("test-model:vllm")

        # Should be sorted alphabetically
        assert endpoints == [
            "192.168.1.10:8080",
            "192.168.1.20:8080",
            "192.168.1.30:8080",
        ]

        # Verify ordering is consistent across multiple calls
        for _ in range(5):
            assert database.get_deployment_endpoints("test-model:vllm") == [
                "192.168.1.10:8080",
                "192.168.1.20:8080",
                "192.168.1.30:8080",
            ]

    def test_get_all_healthy_endpoints(self, database):
        """Test getting all healthy endpoints grouped by deployment."""
        database.add_deployment_endpoint("model-a:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("model-a:vllm", "192.168.1.11:8080")
        database.add_deployment_endpoint("model-b:vllm", "192.168.1.20:8080")
        database.mark_endpoint_unhealthy("model-a:vllm", "192.168.1.11:8080")

        all_endpoints = database.get_all_healthy_endpoints()

        assert "model-a:vllm" in all_endpoints
        assert "model-b:vllm" in all_endpoints
        assert all_endpoints["model-a:vllm"] == ["192.168.1.10:8080"]
        assert all_endpoints["model-b:vllm"] == ["192.168.1.20:8080"]


class TestDatabaseSchema:
    """Tests for database schema and migrations."""

    def test_schema_version(self, database):
        """Test that schema version is tracked correctly."""
        # Schema version is tracked via the schema_version table
        conn = database._get_connection()
        cursor = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] >= 3  # v3 is the new schema

    def test_wal_mode_enabled(self, database):
        """Test that WAL mode is enabled."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_deployments_table_exists(self, database):
        """Test that deployments table exists with correct columns."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA table_info(deployments)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "id",
            "model_name",
            "backend",
            "status",
            "desired_replicas",
            "min_replicas",
            "max_replicas",
            "target_pending_requests",
            "keep_alive_seconds",
            "backend_config",
            "created_at",
            "updated_at",
        }
        assert expected_columns.issubset(columns)

    def test_node_storage_table_exists(self, database):
        """Test that node_storage table exists with correct columns."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA table_info(node_storage)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "node_name",
            "sllm_store_endpoint",
            "cached_models",
        }
        assert expected_columns.issubset(columns)

    def test_deployment_endpoints_table_exists(self, database):
        """Test that deployment_endpoints table exists with correct columns."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA table_info(deployment_endpoints)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "deployment_id",
            "endpoint",
            "status",
            "added_at",
        }
        assert expected_columns.issubset(columns)

    def test_deployment_endpoints_primary_key(self, database):
        """Test that deployment_endpoints has correct primary key."""
        conn = database._get_connection()

        # Add same endpoint twice - should not create duplicate
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_deployment_endpoint("test-model:vllm", "192.168.1.10:8080")

        cursor = conn.execute(
            "SELECT COUNT(*) FROM deployment_endpoints "
            "WHERE deployment_id = ? AND endpoint = ?",
            ("test-model:vllm", "192.168.1.10:8080"),
        )
        count = cursor.fetchone()[0]
        assert count == 1
