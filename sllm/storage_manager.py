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
import os
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
            mem_pool = os.environ.get("SLLM_STORE_MEM_POOL_SIZE", "")
            mem_pool_arg = f"--mem-pool-size {mem_pool} " if mem_pool else ""
            command = (
                f"sllm-store start "
                f"--port $PORT "
                f"--storage-path {self.storage_path} "
                f"--host 0.0.0.0 "
                f"{mem_pool_arg}"
            ).strip()

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
    # Checkpoint Prefetch
    # -------------------------------------------------------------------------

    async def prefetch_to_cpu(self, model_name: str, node_name: str = None) -> bool:
        """Prefetch model weights into system memory while the current model runs.

        Two-stage prefetch for reliability:
        1. sllm-store gRPC LoadModelAsync: loads weights into the pinned memory
           pool (cudaHostRegister) on the sllm-store server, enabling fast
           DMA-based GPU transfers when the serverless_llm loader is used.
        2. HF-cache page-warm: reads the HuggingFace safetensors files into the
           OS page cache so that vLLM's default loader finds them already in RAM
           instead of fetching from SSD.  This produces a measurable reduction in
           model-switch latency because the SSD I/O is overlapped with the
           previous group's inference rather than serialised on the critical path.

        Args:
            model_name: Model name (e.g. "Qwen/Qwen3-8B").
            node_name:  Target node. If None, uses first available node.

        Returns:
            True if at least one prefetch path succeeded, False otherwise.
        """
        loop = asyncio.get_running_loop()
        try:
            # ── HF-cache page-warm (synchronous read, ~4-5 s for 8B) ──
            # vLLM uses load_format=auto, loading safetensors from
            # ~/.cache/huggingface/hub.  Reading those files here warms the
            # OS page cache so that vLLM's weight-load step at model-switch
            # time hits RAM instead of SSD.  We use HF-cache warm exclusively
            # rather than sllm-store gRPC to avoid double NVMe I/O (sllm-store
            # reads tensor.data from the same SSD, which would saturate
            # bandwidth and defeat the purpose).
            hf_result = await loop.run_in_executor(
                None,
                self._prefetch_hf_cache,
                model_name,
            )
            if hf_result:
                return True

            # ── Fallback: sllm-store pinned-memory prefetch ──
            # Used if no HF cache is present (e.g., custom model or offline).
            endpoint = None
            if node_name:
                endpoint = await self.get_store_endpoint(node_name)
            if not endpoint:
                for ep in self._store_endpoints.values():
                    endpoint = ep
                    break

            if endpoint:
                return await loop.run_in_executor(
                    None,
                    self._prefetch_via_store,
                    model_name,
                    endpoint,
                )

            # Last resort: raw read of sllm-store checkpoint files
            logger.warning(
                f"[PREFETCH] No HF cache or sllm-store endpoint, "
                f"falling back to raw file read for {model_name}"
            )
            return await loop.run_in_executor(
                None,
                self._prefetch_raw_read,
                model_name,
            )
        except Exception as e:
            logger.error(f"[PREFETCH] Error prefetching {model_name}: {e}")
            return False

    # --- synchronous helpers (executed in thread-pool) ----------------------

    @staticmethod
    def _prefetch_via_store(model_name: str, endpoint: str) -> bool:
        """Load model into pinned CPU memory via sllm-store gRPC."""
        import time

        t0 = time.monotonic()
        logger.info(
            f"[PREFETCH] Loading {model_name} into pinned memory "
            f"via sllm-store at {endpoint}"
        )

        try:
            from sllm_store.client import SllmStoreClient

            client = SllmStoreClient(server_address=endpoint)
            response = client.load_into_cpu(model_name)

            elapsed = time.monotonic() - t0
            if response is False:
                logger.error(
                    f"[PREFETCH] sllm-store failed to load {model_name} "
                    f"after {elapsed:.1f}s"
                )
                return False

            logger.info(
                f"[PREFETCH] {model_name} loaded into pinned memory "
                f"via sllm-store ({elapsed:.1f}s)"
            )
            return True

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                f"[PREFETCH] sllm-store prefetch failed for {model_name} "
                f"after {elapsed:.1f}s: {e}"
            )
            return False

    async def unload_from_cpu(self, model_name: str) -> bool:
        """Unload model from sllm-store's pinned CPU memory.

        Used by the batch scheduler to proactively free pool space after a
        model group finishes and no future tasks need the model.
        """
        loop = asyncio.get_running_loop()
        try:
            endpoint = None
            for ep in self._store_endpoints.values():
                endpoint = ep
                break

            if not endpoint:
                logger.warning(
                    f"[UNLOAD] No sllm-store endpoint available, "
                    f"cannot unload {model_name}"
                )
                return False

            result = await loop.run_in_executor(
                None,
                self._unload_via_store,
                model_name,
                endpoint,
            )
            return result
        except Exception as e:
            logger.error(f"[UNLOAD] Error unloading {model_name}: {e}")
            return False

    @staticmethod
    def _unload_via_store(model_name: str, endpoint: str) -> bool:
        """Unload model from pinned CPU memory via sllm-store gRPC."""
        try:
            from sllm_store.client import SllmStoreClient

            client = SllmStoreClient(server_address=endpoint)
            response = client.unload_from_cpu(model_name)

            if response is False:
                logger.error(f"[UNLOAD] sllm-store failed to unload {model_name}")
                return False

            logger.info(f"[UNLOAD] {model_name} removed from pinned memory")
            return True
        except Exception as e:
            logger.error(f"[UNLOAD] sllm-store unload failed for {model_name}: {e}")
            return False

    def _prefetch_raw_read(self, model_name: str) -> bool:
        """Fallback: read weight files into OS page cache via raw I/O."""
        import os
        import time

        t0 = time.monotonic()
        model_dir = os.path.join(self.storage_path, model_name)
        logger.info(f"[PREFETCH] Raw-reading weight files from {model_dir}")

        if not os.path.isdir(model_dir):
            logger.error(f"[PREFETCH] Model directory not found: {model_dir}")
            return False

        try:
            total_bytes = 0
            for fname in os.listdir(model_dir):
                if fname.endswith((".safetensors", ".bin")):
                    fpath = os.path.join(model_dir, fname)
                    with open(fpath, "rb") as f:
                        while chunk := f.read(64 * 1024 * 1024):  # 64 MB
                            total_bytes += len(chunk)

            elapsed = time.monotonic() - t0
            total_gb = total_bytes / (1024 ** 3)
            logger.info(
                f"[PREFETCH] Raw-read {model_name} into page cache "
                f"({total_gb:.1f} GB, {elapsed:.1f}s)"
            )
            return True

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                f"[PREFETCH] Raw-read failed for {model_name} "
                f"after {elapsed:.1f}s: {e}"
            )
            return False

    @staticmethod
    def _prefetch_hf_cache(model_name: str) -> bool:
        """Warm OS page cache for model's HuggingFace checkpoint files.

        vLLM uses load_format=auto by default, which reads safetensors shards
        from ~/.cache/huggingface/hub/.  Reading those files here, while the
        previous model group is still executing, moves the SSD→RAM transfer
        off the critical path: when vLLM loads the model at switch time it
        finds the data already in RAM (page cache hit, <0.1 s) instead of
        reading from SSD (~4-5 s for an 8B model).

        Eviction via posix_fadvise(DONTNEED) must be called before each
        benchmark run (see evict_models_page_cache in exp_common.py) to ensure
        this read represents a genuine cold-disk access.
        """
        import os
        import time

        t0 = time.monotonic()
        # Locate HF snapshot directory: Qwen/Qwen3-8B → models--Qwen--Qwen3-8B
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        hf_dir = "models--" + model_name.replace("/", "--")
        snapshots_dir = os.path.join(hf_cache, hf_dir, "snapshots")

        if not os.path.isdir(snapshots_dir):
            logger.warning(
                f"[PREFETCH-HF] No HF cache found for {model_name} "
                f"at {snapshots_dir} — skipping page-warm"
            )
            return False

        # Use the first (newest) snapshot directory
        snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
        if not snapshots:
            return False
        snap_path = os.path.join(snapshots_dir, snapshots[0])

        total_bytes = 0
        try:
            weight_files = sorted(
                f for f in os.listdir(snap_path)
                if f.endswith((".safetensors", ".bin", ".pt"))
            )
            if not weight_files:
                logger.warning(
                    f"[PREFETCH-HF] No weight files in {snap_path}"
                )
                return False

            for fname in weight_files:
                fpath = os.path.join(snap_path, fname)
                with open(fpath, "rb") as f:
                    while chunk := f.read(64 * 1024 * 1024):  # 64 MB chunks
                        total_bytes += len(chunk)

            elapsed = time.monotonic() - t0
            total_gb = total_bytes / (1024 ** 3)
            logger.info(
                f"[PREFETCH-HF] {model_name} page-warmed "
                f"({total_gb:.1f} GB in {elapsed:.1f}s, "
                f"{total_gb / elapsed:.1f} GB/s)"
            )
            return True

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                f"[PREFETCH-HF] Failed for {model_name} "
                f"after {elapsed:.1f}s: {e}"
            )
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
