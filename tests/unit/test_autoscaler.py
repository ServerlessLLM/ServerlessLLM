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
Tests for the Autoscaler component.

These tests cover both the existing autoscaler functionality and the new
receive_metrics method used by the Router.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================ #
# Autoscaler receive_metrics Tests (New - for Router integration)
# ============================================================================ #


class TestAutoscalerReceiveMetrics:
    """Tests for Autoscaler.receive_metrics() method used by Router."""

    def test_receive_metrics_stores_buffer_length(
        self, autoscaler_with_metrics
    ):
        """Test that receive_metrics stores buffer length."""
        autoscaler_with_metrics.receive_metrics(
            model_id="test-model:vllm",
            buffer_len=5,
            in_flight=0,
        )

        metrics = autoscaler_with_metrics.get_metrics("test-model:vllm")
        assert metrics["buffer_len"] == 5

    def test_receive_metrics_stores_in_flight(self, autoscaler_with_metrics):
        """Test that receive_metrics stores in-flight count."""
        autoscaler_with_metrics.receive_metrics(
            model_id="test-model:vllm",
            buffer_len=0,
            in_flight=3,
        )

        metrics = autoscaler_with_metrics.get_metrics("test-model:vllm")
        assert metrics["in_flight"] == 3

    def test_receive_metrics_updates_existing(self, autoscaler_with_metrics):
        """Test that receive_metrics updates existing metrics."""
        autoscaler_with_metrics.receive_metrics(
            model_id="test-model:vllm",
            buffer_len=5,
            in_flight=2,
        )
        autoscaler_with_metrics.receive_metrics(
            model_id="test-model:vllm",
            buffer_len=3,
            in_flight=4,
        )

        metrics = autoscaler_with_metrics.get_metrics("test-model:vllm")
        assert metrics["buffer_len"] == 3
        assert metrics["in_flight"] == 4

    def test_receive_metrics_isolated_per_model(self, autoscaler_with_metrics):
        """Test that metrics are isolated per model."""
        autoscaler_with_metrics.receive_metrics(
            model_id="model-a:vllm",
            buffer_len=5,
            in_flight=2,
        )
        autoscaler_with_metrics.receive_metrics(
            model_id="model-b:vllm",
            buffer_len=10,
            in_flight=0,
        )

        metrics_a = autoscaler_with_metrics.get_metrics("model-a:vllm")
        metrics_b = autoscaler_with_metrics.get_metrics("model-b:vllm")

        assert metrics_a["buffer_len"] == 5
        assert metrics_b["buffer_len"] == 10

    def test_get_total_demand_uses_pushed_metrics(
        self, autoscaler_with_metrics
    ):
        """Test that total demand calculation uses pushed metrics."""
        autoscaler_with_metrics.receive_metrics(
            model_id="test-model:vllm",
            buffer_len=5,
            in_flight=3,
        )

        total_demand = autoscaler_with_metrics.get_total_demand(
            "test-model:vllm"
        )
        assert total_demand == 8  # 5 + 3

    def test_get_total_demand_unknown_model_returns_zero(
        self, autoscaler_with_metrics
    ):
        """Test that unknown model returns zero demand."""
        total_demand = autoscaler_with_metrics.get_total_demand("unknown:vllm")
        assert total_demand == 0


# ============================================================================ #
# Autoscaler Scaling Logic Tests
# ============================================================================ #


class TestAutoscalerScaling:
    """Tests for Autoscaler scaling logic."""

    @pytest.mark.asyncio
    async def test_scale_up_when_demand_increases(
        self, autoscaler_with_metrics, database, sample_model
    ):
        """Test that autoscaler scales up when demand increases."""
        # Push metrics indicating demand
        autoscaler_with_metrics.receive_metrics(
            model_id=sample_model.id,
            buffer_len=15,
            in_flight=5,
        )

        # Run one scaling cycle
        await autoscaler_with_metrics._scale_model(sample_model)

        # Check desired replicas was updated
        updated_model = database.get_model(sample_model.id)
        assert updated_model.desired_replicas > 0

    @pytest.mark.asyncio
    async def test_scale_down_when_no_demand(
        self, autoscaler_with_metrics, database, sample_model
    ):
        """Test that autoscaler scales down when no demand."""
        # First scale up
        database.update_desired_replicas(sample_model.id, 2)

        # Push zero metrics
        autoscaler_with_metrics.receive_metrics(
            model_id=sample_model.id,
            buffer_len=0,
            in_flight=0,
        )

        # Disable keep_alive for this test
        database._get_connection().execute(
            "UPDATE models SET keep_alive_seconds = 0 WHERE id = ?",
            (sample_model.id,),
        )

        # Refresh model
        sample_model = database.get_model(sample_model.id)

        # Run scaling cycle
        await autoscaler_with_metrics._scale_model(sample_model)

        # Check desired replicas was reduced
        updated_model = database.get_model(sample_model.id)
        assert updated_model.desired_replicas == sample_model.min_replicas

    @pytest.mark.asyncio
    async def test_respects_min_replicas(
        self, autoscaler_with_metrics, database
    ):
        """Test that autoscaler respects min_replicas."""
        # Create model with min_replicas=1
        model = database.create_model(
            model_id="min-replica-test:vllm",
            model_name="min-replica-test",
            backend="vllm",
            min_replicas=1,
            max_replicas=4,
        )

        # Push zero demand
        autoscaler_with_metrics.receive_metrics(
            model_id=model.id,
            buffer_len=0,
            in_flight=0,
        )

        # Run scaling
        await autoscaler_with_metrics._scale_model(model)

        # Should not go below min_replicas
        updated_model = database.get_model(model.id)
        assert updated_model.desired_replicas >= 1

    @pytest.mark.asyncio
    async def test_respects_max_replicas(
        self, autoscaler_with_metrics, database
    ):
        """Test that autoscaler respects max_replicas."""
        # Create model with max_replicas=2
        model = database.create_model(
            model_id="max-replica-test:vllm",
            model_name="max-replica-test",
            backend="vllm",
            min_replicas=0,
            max_replicas=2,
            target_pending_requests=1,  # Very low to trigger high scaling
        )

        # Push high demand
        autoscaler_with_metrics.receive_metrics(
            model_id=model.id,
            buffer_len=100,
            in_flight=50,
        )

        # Run scaling
        await autoscaler_with_metrics._scale_model(model)

        # Should not exceed max_replicas
        updated_model = database.get_model(model.id)
        assert updated_model.desired_replicas <= 2


# ============================================================================ #
# Autoscaler Keep-Alive Tests
# ============================================================================ #


class TestAutoscalerKeepAlive:
    """Tests for Autoscaler keep-alive functionality."""

    @pytest.mark.asyncio
    async def test_keep_alive_prevents_immediate_scale_down(
        self, autoscaler_with_metrics, database
    ):
        """Test that keep_alive prevents immediate scale down."""
        # Create model with keep_alive
        model = database.create_model(
            model_id="keepalive-test:vllm",
            model_name="keepalive-test",
            backend="vllm",
            min_replicas=0,
            max_replicas=2,
            keep_alive_seconds=60,
        )
        database.update_desired_replicas(model.id, 1)

        # Push zero demand
        autoscaler_with_metrics.receive_metrics(
            model_id=model.id,
            buffer_len=0,
            in_flight=0,
        )

        # Refresh model
        model = database.get_model(model.id)

        # Run scaling - should NOT scale down due to keep_alive
        await autoscaler_with_metrics._scale_model(model)

        # Should still have 1 replica
        updated_model = database.get_model(model.id)
        assert updated_model.desired_replicas == 1
