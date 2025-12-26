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
Reconciler for ServerlessLLM v1-beta.

Declarative reconciliation loop that:
- Reads desired state from SQLite (desired_replicas)
- Queries Pylet for current instances
- Creates/deletes instances to match desired state
- Health checks starting instances
- Updates LB endpoints

Follows the Kubernetes reconciliation pattern.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import aiohttp

from sllm.command_builder import build_sglang_command, build_vllm_command
from sllm.database import Database, Model
from sllm.lb_registry import LoadBalancerRegistry
from sllm.logger import init_logger
from sllm.pylet_client import InstanceInfo, PyletClient
from sllm.storage_manager import StorageManager

logger = init_logger(__name__)

# Reconciler configuration
RECONCILE_INTERVAL_SECONDS = 3
HEALTH_CHECK_TIMEOUT_SECONDS = 5
STARTUP_TIMEOUT_SECONDS = 300


@dataclass
class ReconcileState:
    """State of instances for a model."""

    ready: List[InstanceInfo]  # RUNNING and in LB
    starting: List[InstanceInfo]  # PENDING, ASSIGNED, or RUNNING but not in LB
    failed: List[InstanceInfo]  # FAILED, COMPLETED, UNKNOWN


class Reconciler:
    """
    Declarative reconciliation loop for model instances.

    The reconciler continuously ensures that the actual state (instances
    running via Pylet) matches the desired state (desired_replicas in SQLite).
    """

    def __init__(
        self,
        database: Database,
        pylet_client: PyletClient,
        lb_registry: LoadBalancerRegistry,
        storage_manager: StorageManager,
        storage_path: str = "/models",
    ):
        """
        Initialize the Reconciler.

        Args:
            database: Database instance
            pylet_client: Pylet client instance
            lb_registry: LoadBalancer registry
            storage_manager: StorageManager instance
            storage_path: Path to model storage on workers
        """
        self.database = database
        self.pylet_client = pylet_client
        self.lb_registry = lb_registry
        self.storage_manager = storage_manager
        self.storage_path = storage_path

        # Track instance startup times for timeout detection
        self._startup_times: Dict[str, datetime] = {}

        # Shutdown flag
        self._shutdown = False

        # HTTP session for health checks
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Start the reconciler background loop."""
        self._session = aiohttp.ClientSession()
        logger.info("Reconciler started")

    async def stop(self):
        """Stop the reconciler."""
        self._shutdown = True
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Reconciler stopped")

    async def run(self):
        """Main reconciliation loop."""
        logger.info("Reconciler loop starting")

        while not self._shutdown:
            try:
                await self._reconcile_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconcile loop: {e}", exc_info=True)

            await asyncio.sleep(RECONCILE_INTERVAL_SECONDS)

        logger.info("Reconciler loop stopped")

    async def _reconcile_all(self):
        """Reconcile all active models."""
        models = self.database.get_active_models()

        for model in models:
            try:
                await self._reconcile_model(model)
            except Exception as e:
                logger.error(
                    f"Error reconciling {model.id}: {e}", exc_info=True
                )

        # Also clean up models marked for deletion
        deleting_models = [
            m for m in self.database.get_all_models() if m.status == "deleting"
        ]
        for model in deleting_models:
            try:
                await self._cleanup_deleting_model(model)
            except Exception as e:
                logger.error(
                    f"Error cleaning up {model.id}: {e}", exc_info=True
                )

    async def _reconcile_model(self, model: Model):
        """
        Reconcile a single model.

        Args:
            model: Model to reconcile
        """
        desired = model.desired_replicas

        # Get current state from Pylet
        instances = await self.pylet_client.get_model_instances(model.id)

        # Get LoadBalancer
        lb = self.lb_registry.get(model.id)
        if not lb:
            logger.warning(f"No LoadBalancer for {model.id}")
            return

        # Categorize instances
        state = self._categorize_instances(instances, lb)

        # Log state
        logger.debug(
            f"[{model.id}] desired={desired}, ready={len(state.ready)}, "
            f"starting={len(state.starting)}, failed={len(state.failed)}"
        )

        # 1. Clean up failed instances
        for inst in state.failed:
            await self._cleanup_instance(inst, lb)

        # 2. Health check starting instances
        for inst in list(state.starting):
            if inst.status == "RUNNING":
                if await self._is_healthy(inst.endpoint):
                    await lb.add_endpoint(inst.endpoint)
                    state.starting.remove(inst)
                    state.ready.append(inst)
                    self._startup_times.pop(inst.instance_id, None)
                    logger.info(
                        f"[{model.id}] Instance {inst.instance_id} is now ready"
                    )
                elif self._is_startup_timeout(inst.instance_id):
                    # Startup timeout - clean up
                    logger.warning(
                        f"[{model.id}] Instance {inst.instance_id} startup timeout"
                    )
                    await self._cleanup_instance(inst, lb)
                    state.starting.remove(inst)

        # 3. Scale up if needed
        current_or_starting = len(state.ready) + len(state.starting)
        need = desired - current_or_starting

        if need > 0:
            logger.info(f"[{model.id}] Scaling up: need {need} more instances")
            for _ in range(need):
                await self._create_instance(model, state.ready + state.starting)

        # 4. Scale down if needed (only from ready, not starting)
        excess = len(state.ready) - desired
        if excess > 0 and len(state.starting) == 0:
            logger.info(f"[{model.id}] Scaling down: {excess} excess instances")
            # Prefer to remove instances NOT on nodes with cached models
            instances_to_remove = self._select_instances_to_remove(
                model.model_name, state.ready, excess
            )
            for inst in instances_to_remove:
                await self._remove_instance(inst, lb)

    def _categorize_instances(
        self,
        instances: List[InstanceInfo],
        lb,
    ) -> ReconcileState:
        """
        Categorize instances into ready, starting, and failed.

        Args:
            instances: List of instances from Pylet
            lb: LoadBalancer for checking endpoints

        Returns:
            ReconcileState with categorized instances
        """
        ready = []
        starting = []
        failed = []

        for inst in instances:
            if inst.status == "RUNNING":
                if lb.has_endpoint(inst.endpoint):
                    ready.append(inst)
                else:
                    starting.append(inst)
                    # Track startup time for timeout
                    if inst.instance_id not in self._startup_times:
                        self._startup_times[inst.instance_id] = datetime.now(
                            timezone.utc
                        )
            elif inst.status in ("PENDING", "ASSIGNED"):
                starting.append(inst)
                if inst.instance_id not in self._startup_times:
                    self._startup_times[inst.instance_id] = datetime.now(
                        timezone.utc
                    )
            else:
                # FAILED, COMPLETED, UNKNOWN, CANCELLED
                failed.append(inst)

        return ReconcileState(ready=ready, starting=starting, failed=failed)

    async def _is_healthy(self, endpoint: str) -> bool:
        """
        Check if an instance is healthy.

        Args:
            endpoint: Instance endpoint (ip:port)

        Returns:
            True if healthy
        """
        if not endpoint or not self._session:
            return False

        try:
            url = f"http://{endpoint}/health"
            async with self._session.get(
                url,
                timeout=aiohttp.ClientTimeout(
                    total=HEALTH_CHECK_TIMEOUT_SECONDS
                ),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _is_startup_timeout(self, instance_id: str) -> bool:
        """Check if instance has exceeded startup timeout."""
        start_time = self._startup_times.get(instance_id)
        if not start_time:
            return False

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        return elapsed > STARTUP_TIMEOUT_SECONDS

    async def _create_instance(
        self,
        model: Model,
        existing: List[InstanceInfo],
    ):
        """
        Create a new instance for a model.

        Args:
            model: Model to create instance for
            existing: Existing instances for storage-aware placement
        """
        # Get backend config
        backend_config = model.backend_config or {}
        tp = backend_config.get("tensor_parallel_size", 1)

        # Select best node
        node = await self.storage_manager.select_best_node(
            model.model_name, tp, existing
        )
        if not node:
            logger.warning(f"[{model.id}] No suitable node for new instance")
            return

        # Ensure sllm-store is running on node
        store_endpoint = await self.storage_manager.ensure_store_on_node(node)
        if not store_endpoint:
            logger.warning(f"[{model.id}] Failed to start sllm-store on {node}")
            return

        # Select GPU indices
        gpu_indices = await self.storage_manager.select_gpu_indices(node, tp)
        if not gpu_indices:
            logger.warning(f"[{model.id}] Not enough GPUs on {node}")
            return

        # Build command
        if model.backend == "sglang":
            command = build_sglang_command(model, self.storage_path)
        else:
            command = build_vllm_command(model, self.storage_path)

        # Create instance via Pylet
        try:
            import uuid

            safe_model = model.model_name.replace("/", "-")
            instance_name = f"{safe_model}-{uuid.uuid4().hex[:8]}"

            instance = await self.pylet_client.submit(
                command=command,
                name=instance_name,
                target_worker=node,
                gpu_indices=gpu_indices,
                exclusive=True,
                labels={
                    "model_id": model.id,
                    "type": "inference",
                    "node": node,
                },
                env={
                    "STORAGE_PATH": self.storage_path,
                    "SLLM_STORE_ENDPOINT": store_endpoint,
                },
            )

            logger.info(
                f"[{model.id}] Created instance {instance.instance_id} "
                f"on {node} with GPUs {gpu_indices}"
            )

        except Exception as e:
            logger.error(f"[{model.id}] Failed to create instance: {e}")

    async def _cleanup_instance(self, inst: InstanceInfo, lb):
        """Clean up a failed or timed-out instance."""
        # Remove from LB
        if inst.endpoint:
            await lb.remove_endpoint(inst.endpoint)

        # Cancel in Pylet
        try:
            await self.pylet_client.cancel_instance(inst.instance_id)
        except Exception as e:
            logger.warning(f"Failed to cancel instance {inst.instance_id}: {e}")

        # Clean up tracking
        self._startup_times.pop(inst.instance_id, None)

        logger.info(f"Cleaned up instance {inst.instance_id}")

    async def _remove_instance(self, inst: InstanceInfo, lb):
        """Remove an instance during scale-down with graceful draining.

        Removes endpoint from LB first (no new requests), waits briefly
        for in-flight requests to complete, then cancels the instance.
        """
        # Remove from LB first (prevents new requests)
        if inst.endpoint and lb:
            await lb.remove_endpoint(inst.endpoint)

            # Wait briefly for in-flight requests to complete
            # A more sophisticated approach would track per-endpoint
            # request counts, but this provides basic graceful draining
            await asyncio.sleep(2.0)

        # Cancel in Pylet
        try:
            await self.pylet_client.cancel_instance(inst.instance_id)
        except Exception as e:
            logger.warning(f"Failed to cancel instance {inst.instance_id}: {e}")

        logger.info(f"Removed instance {inst.instance_id}")

    def _select_instances_to_remove(
        self,
        model_name: str,
        ready: List[InstanceInfo],
        count: int,
    ) -> List[InstanceInfo]:
        """
        Select instances to remove during scale-down.

        Prefers removing instances NOT on nodes with cached models.

        Args:
            model_name: Model name
            ready: Ready instances
            count: Number to remove

        Returns:
            Instances to remove
        """
        # Get nodes with cached model
        nodes_with_cache = set(
            self.storage_manager.get_nodes_with_model(model_name)
        )

        # Sort: instances NOT on cached nodes first
        sorted_instances = sorted(
            ready,
            key=lambda i: (
                i.labels.get("node", "") in nodes_with_cache,
                i.instance_id,
            ),
        )

        return sorted_instances[:count]

    async def _cleanup_deleting_model(self, model: Model):
        """Clean up a model marked for deletion."""
        instances = await self.pylet_client.get_model_instances(model.id)

        if not instances:
            # All instances gone, can delete from database
            self.database.delete_model(model.id)
            logger.info(f"Completed deletion of {model.id}")
            return

        # Cancel remaining instances
        lb = self.lb_registry.get(model.id)
        for inst in instances:
            await self._cleanup_instance(inst, lb if lb else None)


# Global instance
_reconciler: Optional[Reconciler] = None


def get_reconciler() -> Optional[Reconciler]:
    """Get the global Reconciler instance."""
    return _reconciler


def init_reconciler(
    database: Database,
    pylet_client: PyletClient,
    lb_registry: LoadBalancerRegistry,
    storage_manager: StorageManager,
    storage_path: str = "/models",
) -> Reconciler:
    """Initialize the global Reconciler instance."""
    global _reconciler
    _reconciler = Reconciler(
        database=database,
        pylet_client=pylet_client,
        lb_registry=lb_registry,
        storage_manager=storage_manager,
        storage_path=storage_path,
    )
    return _reconciler
