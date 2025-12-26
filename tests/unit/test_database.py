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


class TestDatabase:
    """Tests for Database class."""

    def test_create_model(self, database, sample_model_data):
        """Test model creation."""
        model = database.create_model(**sample_model_data)

        assert model.id == "facebook/opt-125m:vllm"
        assert model.model_name == "facebook/opt-125m"
        assert model.backend == "vllm"
        assert model.status == "active"
        assert model.desired_replicas == 0
        assert model.min_replicas == 0
        assert model.max_replicas == 4
        assert model.target_pending_requests == 10
        assert model.keep_alive_seconds == 60
        assert model.backend_config["tensor_parallel_size"] == 1

    def test_get_model(self, database, sample_model_data):
        """Test getting a model by ID."""
        created = database.create_model(**sample_model_data)
        retrieved = database.get_model(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.model_name == created.model_name

    def test_get_model_not_found(self, database):
        """Test getting a non-existent model."""
        model = database.get_model("nonexistent:vllm")
        assert model is None

    def test_get_all_models(self, database):
        """Test getting all models."""
        # Create multiple models
        database.create_model(
            model_id="model1:vllm",
            model_name="model1",
            backend="vllm",
            min_replicas=0,
            max_replicas=2,
        )
        database.create_model(
            model_id="model2:sglang",
            model_name="model2",
            backend="sglang",
            min_replicas=1,
            max_replicas=4,
        )

        models = database.get_all_models()
        assert len(models) == 2
        assert {m.id for m in models} == {"model1:vllm", "model2:sglang"}

    def test_get_active_models(self, database):
        """Test getting only active models."""
        # Create models with different statuses
        database.create_model(
            model_id="active1:vllm", model_name="active1", backend="vllm"
        )
        database.create_model(
            model_id="active2:vllm", model_name="active2", backend="vllm"
        )

        # Update one to deleting status
        conn = database._get_connection()
        conn.execute(
            "UPDATE models SET status = ? WHERE id = ?",
            ("deleting", "active2:vllm"),
        )

        active_models = database.get_active_models()
        assert len(active_models) == 1
        assert active_models[0].id == "active1:vllm"

    def test_update_desired_replicas(self, database, sample_model):
        """Test updating desired replicas."""
        database.update_desired_replicas(sample_model.id, 3)

        updated = database.get_model(sample_model.id)
        assert updated.desired_replicas == 3

    def test_update_model_status(self, database, sample_model):
        """Test updating model status."""
        database.update_model_status(sample_model.id, "deleting")

        updated = database.get_model(sample_model.id)
        assert updated.status == "deleting"

    def test_delete_model(self, database, sample_model):
        """Test deleting a model."""
        database.delete_model(sample_model.id)

        deleted = database.get_model(sample_model.id)
        assert deleted is None

    def test_duplicate_model_raises_error(self, database, sample_model_data):
        """Test that creating a duplicate model raises an error."""
        database.create_model(**sample_model_data)

        with pytest.raises(ValueError):
            database.create_model(**sample_model_data)


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


class TestModelEndpoints:
    """Tests for model_endpoints table operations."""

    def test_add_model_endpoint(self, database):
        """Test adding an endpoint for a model."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")

        endpoints = database.get_model_endpoints("test-model:vllm")
        assert endpoints == ["192.168.1.10:8080"]

    def test_add_multiple_endpoints(self, database):
        """Test adding multiple endpoints for a model."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.11:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.12:8080")

        endpoints = database.get_model_endpoints("test-model:vllm")
        assert len(endpoints) == 3
        assert set(endpoints) == {
            "192.168.1.10:8080",
            "192.168.1.11:8080",
            "192.168.1.12:8080",
        }

    def test_add_duplicate_endpoint_updates(self, database):
        """Test that adding duplicate endpoint updates instead of creating new."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")

        endpoints = database.get_model_endpoints("test-model:vllm")
        assert len(endpoints) == 1

    def test_remove_model_endpoint(self, database):
        """Test removing an endpoint."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.11:8080")

        database.remove_model_endpoint("test-model:vllm", "192.168.1.10:8080")

        endpoints = database.get_model_endpoints("test-model:vllm")
        assert endpoints == ["192.168.1.11:8080"]

    def test_remove_nonexistent_endpoint(self, database):
        """Test that removing non-existent endpoint doesn't raise."""
        # Should not raise
        database.remove_model_endpoint("test-model:vllm", "192.168.1.99:8080")

    def test_mark_endpoint_unhealthy(self, database):
        """Test marking endpoint as unhealthy."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.11:8080")

        database.mark_endpoint_unhealthy("test-model:vllm", "192.168.1.10:8080")

        # Only healthy endpoints should be returned
        endpoints = database.get_model_endpoints("test-model:vllm")
        assert endpoints == ["192.168.1.11:8080"]

    def test_get_model_endpoints_empty(self, database):
        """Test getting endpoints for model with no endpoints."""
        endpoints = database.get_model_endpoints("unknown-model:vllm")
        assert endpoints == []

    def test_get_model_endpoints_only_healthy(self, database):
        """Test that only healthy endpoints are returned."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.11:8080")
        database.mark_endpoint_unhealthy("test-model:vllm", "192.168.1.11:8080")

        endpoints = database.get_model_endpoints("test-model:vllm")
        assert len(endpoints) == 1
        assert "192.168.1.10:8080" in endpoints

    def test_delete_model_endpoints(self, database):
        """Test deleting all endpoints for a model."""
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.11:8080")

        database.delete_model_endpoints("test-model:vllm")

        endpoints = database.get_model_endpoints("test-model:vllm")
        assert endpoints == []

    def test_endpoints_isolated_per_model(self, database):
        """Test that endpoints are isolated per model."""
        database.add_model_endpoint("model-a:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("model-b:vllm", "192.168.1.20:8080")

        endpoints_a = database.get_model_endpoints("model-a:vllm")
        endpoints_b = database.get_model_endpoints("model-b:vllm")

        assert endpoints_a == ["192.168.1.10:8080"]
        assert endpoints_b == ["192.168.1.20:8080"]

    def test_get_all_healthy_endpoints(self, database):
        """Test getting all healthy endpoints grouped by model."""
        database.add_model_endpoint("model-a:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("model-a:vllm", "192.168.1.11:8080")
        database.add_model_endpoint("model-b:vllm", "192.168.1.20:8080")
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
        assert row[0] >= 1

    def test_wal_mode_enabled(self, database):
        """Test that WAL mode is enabled."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_models_table_exists(self, database):
        """Test that models table exists with correct columns."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA table_info(models)")
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

    def test_model_endpoints_table_exists(self, database):
        """Test that model_endpoints table exists with correct columns."""
        conn = database._get_connection()
        cursor = conn.execute("PRAGMA table_info(model_endpoints)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "model_id",
            "endpoint",
            "status",
            "added_at",
        }
        assert expected_columns.issubset(columns)

    def test_model_endpoints_primary_key(self, database):
        """Test that model_endpoints has correct primary key."""
        conn = database._get_connection()

        # Add same endpoint twice - should not create duplicate
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")
        database.add_model_endpoint("test-model:vllm", "192.168.1.10:8080")

        cursor = conn.execute(
            "SELECT COUNT(*) FROM model_endpoints WHERE model_id = ? AND endpoint = ?",
            ("test-model:vllm", "192.168.1.10:8080"),
        )
        count = cursor.fetchone()[0]
        assert count == 1
