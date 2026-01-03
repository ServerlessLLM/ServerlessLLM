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
StorageManager for ServerlessLLM v1-beta.

Manages sllm-store lifecycle and aggregates global storage view for
storage-aware scheduling.

Responsibilities:
1. sllm-store lifecycle via Pylet
2. Global cache view (from storage-report pushes)
3. Storage-aware placement scoring
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from sllm.command_builder import VENV_SLLM_STORE
from sllm.database import Database, NodeStorage
from sllm.logger import init_logger
from sllm.pylet_client import InstanceInfo, PyletClient, WorkerInfo

logger = init_logger(__name__)


@dataclass
class StorageReport:
    """Storage report from sllm-store."""

    node_name: str
    sllm_store_endpoint: str
    cached_models: List[str]
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0


class StorageManager:
    """
    Manages sllm-store lifecycle and storage-aware scheduling.

    Note: This component conceptually belongs to sllm-store but temporarily
    resides in SLLM until sllm-store gains cluster awareness.
    """

    def __init__(
        self,
        database: Database,
        pylet_client: PyletClient,
        storage_path: str = "/models",
        head_url: str = "http://localhost:8343",
    ):
        """
        Initialize StorageManager.

        Args:
            database: Database instance for persistence
            pylet_client: Pylet client for instance management
            storage_path: Path to model storage on workers
            head_url: SLLM head URL for sllm-store to report back
        """
        self.database = database
        self.pylet_client = pylet_client
        self.storage_path = storage_path
        self.head_url = head_url

        # In-memory cache view (refreshed from database)
        self._cache_view: Dict[str, Set[str]] = {}  # node_name -> set of models
        self._store_endpoints: Dict[str, str] = {}  # node_name -> endpoint
        self._download_locks: Dict[str, asyncio.Lock] = {}

    async def recover_from_db(self):
        """Recover state from database on startup."""
        logger.info("Recovering StorageManager state from database")

        node_storages = self.database.get_all_node_storage()
        for ns in node_storages:
            self._cache_view[ns.node_name] = set(ns.cached_models)
            if ns.sllm_store_endpoint:
                self._store_endpoints[ns.node_name] = ns.sllm_store_endpoint

        logger.info(f"Recovered storage info for {len(node_storages)} nodes")

    async def initialize(self) -> bool:
        """
        Initialize sllm-store on all online worker nodes.

        This should be called during head node startup to ensure all workers
        have sllm-store running before accepting requests. Starting sllm-store
        is expensive, so doing it eagerly avoids cold-start latency.

        Returns:
            True if all workers initialized successfully, False otherwise
        """
        logger.info("Initializing sllm-store on all worker nodes...")

        # Get all online workers
        workers = await self.pylet_client.get_online_workers()
        if not workers:
            logger.warning("No online workers found during initialization")
            return True  # Not a failure - workers may join later

        logger.info(f"Found {len(workers)} online workers, starting sllm-store")

        # Start sllm-store on all workers in parallel
        tasks = []
        for worker in workers:
            task = asyncio.create_task(
                self._init_store_on_worker(worker),
                name=f"init-store-{worker.worker_id}",
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successes and failures
        success_count = 0
        failure_count = 0
        for worker, result in zip(workers, results):
            if isinstance(result, Exception):
                logger.error(
                    f"Failed to init sllm-store on {worker.worker_id}: {result}"
                )
                failure_count += 1
            elif result:
                success_count += 1
            else:
                failure_count += 1

        logger.info(
            f"sllm-store initialization complete: "
            f"{success_count} succeeded, {failure_count} failed"
        )

        return failure_count == 0

    async def _init_store_on_worker(self, worker: WorkerInfo) -> bool:
        """
        Initialize sllm-store on a single worker.

        Args:
            worker: Worker info

        Returns:
            True if successful
        """
        node_name = worker.worker_id
        try:
            endpoint = await self.ensure_store_on_node(node_name)
            if endpoint:
                logger.info(f"sllm-store ready on {node_name} at {endpoint}")
                return True
            else:
                logger.error(f"sllm-store failed to start on {node_name}")
                return False
        except Exception as e:
            logger.error(f"Error initializing sllm-store on {node_name}: {e}")
            return False

    # -------------------------------------------------------------------------
    # sllm-store Lifecycle
    # -------------------------------------------------------------------------

    async def ensure_store_on_node(self, node_name: str) -> Optional[str]:
        """
        Ensure sllm-store is running on a node.

        Args:
            node_name: Worker node ID

        Returns:
            sllm-store endpoint (ip:port) or None if failed
        """
        # Check if already running
        existing = await self.get_store_endpoint(node_name)
        if existing:
            return existing

        # Check for existing instance in Pylet
        existing_instance = await self.pylet_client.get_store_instance(
            node_name
        )
        if existing_instance and existing_instance.endpoint:
            self._store_endpoints[node_name] = existing_instance.endpoint
            return existing_instance.endpoint

        # Get worker info
        worker = await self.pylet_client.get_worker(node_name)
        if not worker or worker.status != "ONLINE":
            logger.warning(
                f"Worker {node_name} not online, cannot start sllm-store"
            )
            return None

        # Start sllm-store instance
        try:
            command = (
                f"sllm-store start "
                f"--port $PORT "
                f"--storage-path {self.storage_path} "
                f"--host 0.0.0.0"
            )

            instance = await self.pylet_client.submit(
                command=command,
                name=f"sllm-store-{node_name}",
                target_worker=node_name,
                gpu_indices=list(range(worker.total_gpus)),
                exclusive=False,  # Shares GPUs with inference
                labels={
                    "type": "sllm-store",
                    "node": node_name,
                },
                env={
                    "STORAGE_PATH": self.storage_path,
                    "SLLM_HEAD_URL": self.head_url,
                },
                venv=VENV_SLLM_STORE,
            )

            # Wait for it to be running
            instance = await self.pylet_client.wait_instance_running(
                instance.instance_id, timeout=60
            )

            if instance.endpoint:
                self._store_endpoints[node_name] = instance.endpoint
                logger.info(
                    f"Started sllm-store on {node_name} at {instance.endpoint}"
                )
                return instance.endpoint
            else:
                logger.error(
                    f"sllm-store started but no endpoint on {node_name}"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to start sllm-store on {node_name}: {e}")
            return None

    async def get_store_endpoint(self, node_name: str) -> Optional[str]:
        """
        Get sllm-store endpoint for a node.

        Args:
            node_name: Worker node ID

        Returns:
            Endpoint (ip:port) or None if not running
        """
        # Check in-memory cache first
        if node_name in self._store_endpoints:
            return self._store_endpoints[node_name]

        # Check database
        node_storage = self.database.get_node_storage(node_name)
        if node_storage and node_storage.sllm_store_endpoint:
            self._store_endpoints[node_name] = node_storage.sllm_store_endpoint
            return node_storage.sllm_store_endpoint

        # Check Pylet
        endpoint = await self.pylet_client.get_store_endpoint(node_name)
        if endpoint:
            self._store_endpoints[node_name] = endpoint
            return endpoint

        return None

    # -------------------------------------------------------------------------
    # Cache Aggregation
    # -------------------------------------------------------------------------

    async def handle_storage_report(self, report: StorageReport):
        """
        Handle storage report from sllm-store.

        Args:
            report: Storage report from sllm-store
        """
        # Update in-memory cache
        self._cache_view[report.node_name] = set(report.cached_models)
        self._store_endpoints[report.node_name] = report.sllm_store_endpoint

        # Persist to database
        self.database.upsert_node_storage(
            node_name=report.node_name,
            sllm_store_endpoint=report.sllm_store_endpoint,
            cached_models=report.cached_models,
        )

        logger.debug(
            f"Updated storage for {report.node_name}: "
            f"{len(report.cached_models)} models cached"
        )

    def get_cached_models(self, node_name: str) -> List[str]:
        """
        Get list of models cached on a node.

        Args:
            node_name: Worker node ID

        Returns:
            List of cached model names
        """
        return list(self._cache_view.get(node_name, set()))

    def get_nodes_with_model(self, model_name: str) -> List[str]:
        """
        Get list of nodes that have a model cached.

        Args:
            model_name: Model name to look for

        Returns:
            List of node names with model cached
        """
        nodes = []
        for node_name, cached_models in self._cache_view.items():
            if model_name in cached_models:
                nodes.append(node_name)
        return nodes

    # -------------------------------------------------------------------------
    # Storage-Aware Placement
    # -------------------------------------------------------------------------

    def score_node(
        self,
        node_name: str,
        model_name: str,
        existing_instances: List[InstanceInfo],
    ) -> int:
        """
        Score a node for placement of a model instance.

        Higher score = better placement.

        Args:
            node_name: Worker node ID
            model_name: Model to place
            existing_instances: Existing instances for this model

        Returns:
            Placement score
        """
        score = 0

        # +100 if model is cached on this node
        cached_models = self._cache_view.get(node_name, set())
        if model_name in cached_models:
            score += 100

        # -10 per existing instance on this node (spread)
        instances_on_node = sum(
            1 for i in existing_instances if i.labels.get("node") == node_name
        )
        score -= instances_on_node * 10

        return score

    async def select_best_node(
        self,
        model_name: str,
        gpu_count: int,
        existing_instances: List[InstanceInfo],
    ) -> Optional[str]:
        """
        Select the best node for placing a model instance.

        Args:
            model_name: Model to place
            gpu_count: Number of GPUs required
            existing_instances: Existing instances for this model

        Returns:
            Node name or None if no suitable node
        """
        workers = await self.pylet_client.get_online_workers()
        if not workers:
            logger.warning("No online workers available")
            return None

        # Filter workers with enough GPUs
        eligible = [w for w in workers if w.available_gpus >= gpu_count]

        if not eligible:
            logger.warning(f"No workers with {gpu_count} available GPUs")
            return None

        # Score each eligible worker
        scores = {}
        for worker in eligible:
            scores[worker.worker_id] = self.score_node(
                worker.worker_id,
                model_name,
                existing_instances,
            )

        # Select highest scoring node
        best_node = max(scores, key=scores.get)
        logger.info(
            f"Selected node {best_node} for {model_name} "
            f"(score: {scores[best_node]})"
        )

        return best_node

    # -------------------------------------------------------------------------
    # Model Download
    # -------------------------------------------------------------------------

    async def download_model(
        self,
        model_name: str,
        backend: str,
        num_nodes: int = 1,
        retries: int = 2,
    ) -> List[str]:
        lock = self._download_locks.setdefault(model_name, asyncio.Lock())
        async with lock:
            existing_nodes = set(self.get_nodes_with_model(model_name))
            if existing_nodes and num_nodes <= len(existing_nodes):
                return list(existing_nodes)[:num_nodes]

            workers = await self.pylet_client.get_online_workers()
            if not workers:
                logger.warning("No online workers available for model download")
                return list(existing_nodes) if existing_nodes else []

            available = [w for w in workers if w.worker_id not in existing_nodes]
            if not available:
                logger.info(f"Model {model_name} already on all available nodes")
                return list(existing_nodes)

            targets = random.sample(available, min(num_nodes, len(available)))
            target_ids = [w.worker_id for w in targets]
            logger.info(f"Downloading {model_name} to nodes: {target_ids}")

            succeeded = list(existing_nodes)
            for node in target_ids:
                for attempt in range(retries):
                    try:
                        if await self._download_model_on_node(node, model_name, backend):
                            succeeded.append(node)
                            break
                    except Exception as e:
                        logger.warning(
                            f"Download attempt {attempt + 1} failed for "
                            f"{model_name} on {node}: {e}"
                        )
                else:
                    logger.error(f"Failed to download {model_name} to {node}")

            logger.info(f"Downloaded {model_name} to {len(succeeded)} nodes")
            return succeeded

    async def ensure_model_downloaded(
        self, model_name: str, backend: str
    ) -> Optional[str]:
        nodes = await self.download_model(model_name, backend, num_nodes=1)
        return nodes[0] if nodes else None

    async def download_to_nodes(
        self, model_name: str, backend: str, num_nodes: int = 1
    ) -> List[str]:
        return await self.download_model(model_name, backend, num_nodes=num_nodes)

    async def _download_model_on_node(
        self, node_name: str, model_name: str, backend: str
    ) -> bool:
        endpoint = await self.ensure_store_on_node(node_name)
        if not endpoint:
            return False

        command = (
            f"sllm-store save "
            f"--model {model_name} "
            f"--backend {backend} "
            f"--storage-path {self.storage_path}"
        )

        import uuid
        safe_model = model_name.replace("/", "-")
        instance = await self.pylet_client.submit(
            command=command,
            name=f"download-{safe_model}-{uuid.uuid4().hex[:8]}",
            target_worker=node_name,
            gpu_indices=[0],
            exclusive=False,
            labels={
                "type": "model-download",
                "model": model_name,
                "node": node_name,
            },
            venv=VENV_SLLM_STORE,
        )

        final = await self.pylet_client.wait_instance_completed(
            instance.instance_id, timeout=600
        )
        if final and final.status == "COMPLETED":
            logger.info(f"Downloaded {model_name} on {node_name}")
            return True
        return False

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def clear_cache_view(self, node_name: str):
        """Clear cache view for a node (e.g., when worker goes offline)."""
        self._cache_view.pop(node_name, None)
        self._store_endpoints.pop(node_name, None)

    def __repr__(self) -> str:
        return (
            f"StorageManager(nodes={len(self._cache_view)}, "
            f"stores={len(self._store_endpoints)})"
        )


# Global instance
_storage_manager: Optional[StorageManager] = None


def get_storage_manager() -> Optional[StorageManager]:
    """Get the global StorageManager instance."""
    return _storage_manager


def init_storage_manager(
    database: Database,
    pylet_client: PyletClient,
    storage_path: str = "/models",
    head_url: str = "http://localhost:8343",
) -> StorageManager:
    """Initialize the global StorageManager instance."""
    global _storage_manager
    _storage_manager = StorageManager(
        database=database,
        pylet_client=pylet_client,
        storage_path=storage_path,
        head_url=head_url,
    )
    return _storage_manager
