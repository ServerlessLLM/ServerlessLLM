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
- Writes to deployment_endpoints table (Router reads from there)

Design (from docs/v1-beta-scalable-router-design.md):
- Endpoint lifecycle: Add/remove endpoints in deployment_endpoints table
- Instance management: Create/delete instances via Pylet
- Health ownership: Mark endpoints healthy/unhealthy based on Pylet heartbeats

Terminology:
- Deployment: A (model_name, backend) pair - the basic scheduling unit
- deployment_id: Unique identifier (format: "{model_name}:{backend}")
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import aiohttp

from sllm.command_builder import build_instance_command
from sllm.database import Database, Deployment
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
    """State of instances for a deployment."""

    ready: List[InstanceInfo]  # RUNNING and in deployment_endpoints
    starting: List[InstanceInfo]  # PENDING, ASSIGNED, or RUNNING but not in DB
    failed: List[InstanceInfo]  # FAILED, COMPLETED, UNKNOWN


class Reconciler:
    """
    Declarative reconciliation loop for deployment instances.

    The reconciler continuously ensures that the actual state (instances
    running via Pylet) matches the desired state (desired_replicas in SQLite).

    Writes to deployment_endpoints table instead of calling LoadBalancer
    directly. The Router reads from this table on every request.
    """

    def __init__(
        self,
        database: Database,
        pylet_client: PyletClient,
        storage_manager: StorageManager,
        storage_path: str = "/models",
    ):
        """
        Initialize the Reconciler.

        Args:
            database: Database instance
            pylet_client: Pylet client instance
            storage_manager: StorageManager instance
            storage_path: Path to model storage on workers
        """
        self.database = database
        self.pylet_client = pylet_client
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
        """Reconcile all active deployments (status='active').

        Skip 'pending', 'downloading', 'failed', 'deleting' deployments.
        """
        deployments = self.database.get_ready_deployments()

        for deployment in deployments:
            try:
                await self._reconcile_deployment(deployment)
            except Exception as e:
                logger.error(
                    f"Error reconciling {deployment.id}: {e}", exc_info=True
                )

        # Also clean up deployments marked for deletion
        deleting_deployments = [
            d
            for d in self.database.get_all_deployments()
            if d.status == "deleting"
        ]
        for deployment in deleting_deployments:
            try:
                await self._cleanup_deleting_deployment(deployment)
            except Exception as e:
                logger.error(
                    f"Error cleaning up {deployment.id}: {e}", exc_info=True
                )

    async def _reconcile_deployment(self, deployment: Deployment):
        """
        Reconcile a single deployment.

        Args:
            deployment: Deployment to reconcile
        """
        desired = deployment.desired_replicas

        # Get current state from Pylet
        instances = await self.pylet_client.get_deployment_instances(
            deployment.id
        )

        # Get endpoints from database
        db_endpoints = set(
            self.database.get_deployment_endpoints(deployment.id)
        )

        # Categorize instances
        state = self._categorize_instances(instances, db_endpoints)

        # Log state
        logger.debug(
            f"[{deployment.id}] desired={desired}, ready={len(state.ready)}, "
            f"starting={len(state.starting)}, failed={len(state.failed)}"
        )

        # 1. Clean up failed instances
        for inst in state.failed:
            await self._cleanup_instance(deployment.id, inst)

        # 2. Health check starting instances
        for inst in list(state.starting):
            if inst.status == "RUNNING":
                if await self._is_healthy(inst.endpoint):
                    # Add to deployment_endpoints table
                    self.database.add_deployment_endpoint(
                        deployment.id, inst.endpoint
                    )
                    state.starting.remove(inst)
                    state.ready.append(inst)
                    self._startup_times.pop(inst.instance_id, None)
                    logger.info(
                        f"[{deployment.id}] Instance {inst.instance_id} is now "
                        f"ready at {inst.endpoint}"
                    )
                elif self._is_startup_timeout(inst.instance_id):
                    # Startup timeout - clean up
                    logger.warning(
                        f"[{deployment.id}] Instance {inst.instance_id} "
                        f"startup timeout"
                    )
                    await self._cleanup_instance(deployment.id, inst)
                    state.starting.remove(inst)

        # 3. Scale up if needed
        current_or_starting = len(state.ready) + len(state.starting)
        need = desired - current_or_starting

        if need > 0:
            logger.info(
                f"[{deployment.id}] Scaling up: need {need} more instances"
            )
            for _ in range(need):
                await self._create_instance(
                    deployment, state.ready + state.starting
                )

        # 4. Scale down if needed (only from ready, not starting)
        excess = len(state.ready) - desired
        if excess > 0 and len(state.starting) == 0:
            logger.info(
                f"[{deployment.id}] Scaling down: {excess} excess instances"
            )
            # Prefer to remove instances NOT on nodes with cached models
            instances_to_remove = self._select_instances_to_remove(
                deployment.model_name, state.ready, excess
            )
            for inst in instances_to_remove:
                await self._remove_instance(deployment.id, inst)

    def _categorize_instances(
        self,
        instances: List[InstanceInfo],
        db_endpoints: Set[str],
    ) -> ReconcileState:
        """
        Categorize instances into ready, starting, and failed.

        Args:
            instances: List of instances from Pylet
            db_endpoints: Set of endpoints in deployment_endpoints table

        Returns:
            ReconcileState with categorized instances
        """
        ready = []
        starting = []
        failed = []

        for inst in instances:
            if inst.status == "RUNNING":
                if inst.endpoint and inst.endpoint in db_endpoints:
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
                # FAILED, COMPLETED, UNKNOWN need cleanup
                # CANCELLED is expected (we requested it during scale-down) - skip
                if inst.status != "CANCELLED":
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
        deployment: Deployment,
        existing: List[InstanceInfo],
    ):
        """
        Create a new instance for a deployment.

        Args:
            deployment: Deployment to create instance for
            existing: Existing instances for storage-aware placement
        """
        backend_config = deployment.backend_config or {}
        tp = backend_config.get("tensor_parallel_size", 1)

        node = await self.storage_manager.ensure_model_on_node(
            deployment.model_name, deployment.backend
        )
        if not node:
            logger.warning(
                f"[{deployment.id}] Model not available, waiting for download"
            )
            return

        # Verify model is actually on node before scheduling (handles cache staleness)
        model_verified = await self.storage_manager.verify_model_on_node(
            node, deployment.model_name
        )
        if not model_verified:
            logger.warning(
                f"[{deployment.id}] Model verification failed on {node}, "
                "will retry next cycle"
            )
            # Clear stale cache entry so next cycle picks a different node or triggers download
            self.storage_manager.clear_cache_view(node)
            return

        # Ensure sllm-store is running on node
        store_endpoint = await self.storage_manager.ensure_store_on_node(node)
        if not store_endpoint:
            logger.warning(
                f"[{deployment.id}] Failed to start sllm-store on {node}"
            )
            return

        # Build command and get venv path
        command, venv_path = build_instance_command(
            deployment, self.storage_path
        )

        # Create instance via Pylet
        # Use gpu=N to let Pylet auto-allocate GPUs
        try:
            import uuid

            safe_model = deployment.model_name.replace("/", "-")
            instance_name = f"{safe_model}-{uuid.uuid4().hex[:8]}"

            instance = await self.pylet_client.submit(
                command=command,
                name=instance_name,
                target_worker=node,
                gpu=tp,  # Let Pylet auto-allocate N GPUs
                exclusive=True,
                labels={
                    "deployment_id": deployment.id,
                    "type": "inference",
                    "node": node,
                },
                env={
                    "STORAGE_PATH": self.storage_path,
                    "SLLM_STORE_ENDPOINT": store_endpoint,
                },
                venv=venv_path,
            )

            logger.info(
                f"[{deployment.id}] Created instance {instance.instance_id} "
                f"on {node} (requested {tp} GPUs)"
            )

        except Exception as e:
            logger.error(f"[{deployment.id}] Failed to create instance: {e}")

    async def _cleanup_instance(self, deployment_id: str, inst: InstanceInfo):
        """Clean up a failed or timed-out instance."""
        # Remove from deployment_endpoints table
        if inst.endpoint:
            self.database.remove_deployment_endpoint(
                deployment_id, inst.endpoint
            )

        # Cancel in Pylet
        try:
            await self.pylet_client.cancel_instance(inst.instance_id)
        except Exception as e:
            logger.warning(f"Failed to cancel instance {inst.instance_id}: {e}")

        # Clean up tracking
        self._startup_times.pop(inst.instance_id, None)

        logger.info(f"Cleaned up instance {inst.instance_id}")

    async def _remove_instance(self, deployment_id: str, inst: InstanceInfo):
        """Remove an instance during scale-down with graceful draining.

        Removes endpoint from deployment_endpoints table first (Router stops
        sending new requests), waits briefly, then cancels the instance.
        """
        # Remove from deployment_endpoints table (Router stops sending requests)
        if inst.endpoint:
            self.database.remove_deployment_endpoint(
                deployment_id, inst.endpoint
            )

            # Wait briefly for in-flight requests to complete
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

    async def _cleanup_deleting_deployment(self, deployment: Deployment):
        """Clean up a deployment marked for deletion."""
        instances = await self.pylet_client.get_deployment_instances(
            deployment.id
        )

        if not instances:
            # All instances gone - remove endpoints and delete from database
            self.database.remove_deployment_endpoints(deployment.id)
            self.database.delete_deployment(deployment.id)
            logger.info(f"Completed deletion of {deployment.id}")
            return

        # Cancel remaining instances
        for inst in instances:
            await self._cleanup_instance(deployment.id, inst)


# Global instance
_reconciler: Optional[Reconciler] = None


def get_reconciler() -> Optional[Reconciler]:
    """Get the global Reconciler instance."""
    return _reconciler


def init_reconciler(
    database: Database,
    pylet_client: PyletClient,
    storage_manager: StorageManager,
    storage_path: str = "/models",
) -> Reconciler:
    """Initialize the global Reconciler instance."""
    global _reconciler
    _reconciler = Reconciler(
        database=database,
        pylet_client=pylet_client,
        storage_manager=storage_manager,
        storage_path=storage_path,
    )
    return _reconciler
