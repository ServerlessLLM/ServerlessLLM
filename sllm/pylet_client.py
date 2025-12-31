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
Pylet client wrapper for ServerlessLLM v1-beta.

Thin wrapper around pylet.aio with SLLM-specific helpers and retry logic.
Pylet is the source of truth for instance state - we query but never duplicate.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sllm.logger import init_logger

logger = init_logger(__name__)

# Pylet import - will fail gracefully if not installed
try:
    import pylet

    PYLET_AVAILABLE = True
except ImportError:
    pylet = None
    PYLET_AVAILABLE = False
    logger.warning(
        "pylet package not installed. Install with: pip install pylet"
    )


@dataclass
class InstanceInfo:
    """SLLM view of a Pylet instance."""

    instance_id: str
    name: Optional[str]
    status: (
        str  # PENDING, ASSIGNED, RUNNING, UNKNOWN, COMPLETED, FAILED, CANCELLED
    )
    endpoint: Optional[str]  # "ip:port" when running
    port: Optional[int]
    gpu_indices: List[int]
    exclusive: bool
    labels: Dict[str, str]
    exit_code: Optional[int]
    failure_reason: Optional[str]

    @property
    def deployment_id(self) -> Optional[str]:
        """Get deployment_id from labels."""
        return self.labels.get("deployment_id")

    @property
    def node(self) -> Optional[str]:
        """Get node from labels."""
        return self.labels.get("node")


@dataclass
class WorkerInfo:
    """SLLM view of a Pylet worker."""

    worker_id: str
    host: str
    status: str  # ONLINE, OFFLINE
    total_gpus: int
    available_gpus: int
    available_gpu_indices: List[int]


class PyletClient:
    """
    Async Pylet client for SLLM.

    Provides SLLM-specific query helpers and retry logic.
    Follows single-source-of-truth principle: we query Pylet, never cache
    instance state locally.
    """

    def __init__(self, endpoint: str = "http://localhost:8000"):
        self.endpoint = endpoint
        self._initialized = False
        self._retry_delay = 1.0
        self._max_retry_delay = 30.0
        self._max_retries = 3

    async def init(self):
        """Initialize connection to Pylet."""
        if not PYLET_AVAILABLE:
            raise RuntimeError(
                "pylet package not installed. Install with: pip install pylet"
            )

        # pylet has a synchronous API - wrap in thread for async context
        await asyncio.to_thread(pylet.init, self.endpoint)
        self._initialized = True
        logger.info(f"Connected to Pylet at {self.endpoint}")

    async def _ensure_initialized(self):
        """Ensure client is initialized."""
        if not self._initialized:
            await self.init()

    async def _with_retry(self, operation, *args, **kwargs):
        """Execute operation with exponential backoff retry."""
        last_error = None
        delay = self._retry_delay

        for attempt in range(self._max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(
                        f"Pylet operation failed (attempt {attempt + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self._max_retry_delay)

        logger.error(
            f"Pylet operation failed after {self._max_retries} retries"
        )
        raise last_error

    # -------------------------------------------------------------------------
    # Instance Operations
    # -------------------------------------------------------------------------

    async def submit(
        self,
        command: str,
        name: Optional[str] = None,
        target_worker: Optional[str] = None,
        gpu: Optional[int] = None,
        gpu_indices: Optional[List[int]] = None,
        exclusive: bool = True,
        labels: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
        venv: Optional[str] = None,
    ) -> InstanceInfo:
        """
        Submit a new instance to Pylet.

        Args:
            command: Command to run
            name: Optional unique name for lookup
            target_worker: Place on specific node
            gpu: Auto-allocate N GPUs (exclusive)
            gpu_indices: Request specific physical GPUs
            exclusive: If False, don't block GPU pool (for sllm-store)
            labels: Custom metadata for filtering
            env: Environment variables
            venv: Path to virtualenv for running the command (pylet 0.4.0+)

        Returns:
            InstanceInfo with instance_id and initial state
        """
        await self._ensure_initialized()

        kwargs = {
            "command": command,
            "exclusive": exclusive,
            "labels": labels or {},
            "env": env or {},
        }

        if name:
            kwargs["name"] = name
        if target_worker:
            kwargs["target_worker"] = target_worker
        if gpu is not None:
            kwargs["gpu"] = gpu
        if gpu_indices is not None:
            kwargs["gpu_indices"] = gpu_indices
        if venv is not None:
            kwargs["venv"] = venv

        # pylet has synchronous API - wrap in thread
        instance = await self._with_retry(
            lambda **kw: asyncio.to_thread(pylet.submit, **kw), **kwargs
        )
        logger.info(f"Submitted instance {instance.id} ({name or 'unnamed'})")

        return self._to_instance_info(instance)

    async def get_instance(self, instance_id: str) -> Optional[InstanceInfo]:
        """Get instance by ID."""
        await self._ensure_initialized()

        try:
            instance = await self._with_retry(
                lambda id: asyncio.to_thread(pylet.get, id=id), instance_id
            )
            return self._to_instance_info(instance)
        except Exception:
            return None

    async def get_instance_by_name(self, name: str) -> Optional[InstanceInfo]:
        """Get instance by name."""
        await self._ensure_initialized()

        try:
            instance = await self._with_retry(
                lambda n: asyncio.to_thread(pylet.get, n), name
            )
            return self._to_instance_info(instance)
        except Exception:
            return None

    async def list_instances(
        self,
        label: Optional[str] = None,
        labels: Optional[List[str]] = None,
        status: Optional[str] = None,
    ) -> List[InstanceInfo]:
        """
        List instances with optional filtering.

        Args:
            label: Single label filter (e.g., "deployment_id:llama-3.1-8b:vllm")
            labels: Multiple label filters
            status: Status filter (e.g., "RUNNING")

        Returns:
            List of matching instances
        """
        await self._ensure_initialized()

        # Convert label filters to dict format expected by pylet
        # pylet.instances takes labels as Dict[str, str], not string filters
        labels_dict = {}
        if label:
            # Parse "key:value" format
            if ":" in label:
                key, value = label.split(":", 1)
                labels_dict[key] = value
        if labels:
            for lbl in labels:
                if ":" in lbl:
                    key, value = lbl.split(":", 1)
                    labels_dict[key] = value

        kwargs = {}
        if labels_dict:
            kwargs["labels"] = labels_dict
        if status:
            kwargs["status"] = status

        # pylet has synchronous API - wrap in thread
        instances = await self._with_retry(
            lambda **kw: asyncio.to_thread(pylet.instances, **kw), **kwargs
        )
        return [self._to_instance_info(inst) for inst in instances]

    async def cancel_instance(self, instance_id: str):
        """Cancel an instance."""
        await self._ensure_initialized()

        instance = await asyncio.to_thread(pylet.get, id=instance_id)
        await asyncio.to_thread(instance.cancel)
        logger.info(f"Cancelled instance {instance_id}")

    async def wait_instance_running(
        self, instance_id: str, timeout: int = 60
    ) -> InstanceInfo:
        """
        Wait for instance to reach RUNNING state.

        Raises TimeoutError if not ready in time.
        """
        await self._ensure_initialized()

        instance = await asyncio.to_thread(pylet.get, id=instance_id)
        await asyncio.to_thread(instance.wait_running, timeout=timeout)
        return self._to_instance_info(instance)

    # -------------------------------------------------------------------------
    # SLLM-Specific Helpers
    # -------------------------------------------------------------------------

    async def get_deployment_instances(
        self, deployment_id: str
    ) -> List[InstanceInfo]:
        """Get all instances for a deployment."""
        return await self.list_instances(label=f"deployment_id:{deployment_id}")

    async def get_running_deployment_instances(
        self, deployment_id: str
    ) -> List[InstanceInfo]:
        """Get running instances for a deployment."""
        return await self.list_instances(
            label=f"deployment_id:{deployment_id}",
            status="RUNNING",
        )

    async def get_store_instance(
        self, node_name: str
    ) -> Optional[InstanceInfo]:
        """Get sllm-store instance on a node."""
        instances = await self.list_instances(
            labels=[f"type:sllm-store", f"node:{node_name}"],
            status="RUNNING",
        )
        return instances[0] if instances else None

    async def get_store_endpoint(self, node_name: str) -> Optional[str]:
        """Get sllm-store endpoint for a node."""
        instance = await self.get_store_instance(node_name)
        return instance.endpoint if instance else None

    # -------------------------------------------------------------------------
    # Worker Operations
    # -------------------------------------------------------------------------

    async def list_workers(self) -> List[WorkerInfo]:
        """List all Pylet workers."""
        await self._ensure_initialized()

        # pylet has synchronous API - wrap in thread
        workers = await self._with_retry(
            lambda: asyncio.to_thread(pylet.workers)
        )
        return [self._to_worker_info(w) for w in workers]

    async def get_online_workers(self) -> List[WorkerInfo]:
        """Get all online workers."""
        workers = await self.list_workers()
        return [w for w in workers if w.status == "ONLINE"]

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get a specific worker by ID."""
        workers = await self.list_workers()
        for w in workers:
            if w.worker_id == worker_id:
                return w
        return None

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def is_healthy(self) -> bool:
        """Check if Pylet connection is healthy."""
        try:
            await self._ensure_initialized()
            # Simple health check: try to list workers
            await asyncio.to_thread(pylet.workers)
            return True
        except Exception as e:
            logger.warning(f"Pylet health check failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _to_instance_info(self, instance) -> InstanceInfo:
        """Convert Pylet instance to InstanceInfo."""
        # pylet 0.3.0 Instance attributes:
        # - id (not instance_id)
        # - name, status, endpoint, gpu_indices, exclusive, labels, exit_code
        # - no port attribute (extract from endpoint)
        # - no failure_reason attribute

        # Extract port from endpoint (format: "host:port")
        port = None
        if instance.endpoint:
            try:
                port = int(instance.endpoint.split(":")[-1])
            except (ValueError, IndexError):
                pass

        return InstanceInfo(
            instance_id=instance.id,
            name=instance.name,
            status=instance.status,
            endpoint=instance.endpoint,
            port=port,
            gpu_indices=list(instance.gpu_indices)
            if instance.gpu_indices
            else [],
            exclusive=instance.exclusive,
            labels=dict(instance.labels) if instance.labels else {},
            exit_code=instance.exit_code,
            failure_reason=None,  # Not available in pylet 0.3.0
        )

    def _to_worker_info(self, worker) -> WorkerInfo:
        """Convert Pylet worker to WorkerInfo."""
        # pylet 0.3.0 WorkerInfo attributes:
        # - id (not worker_id)
        # - host
        # - status
        # - gpu, gpu_available, gpu_indices_available
        # - cpu, cpu_available
        # - memory, memory_available
        return WorkerInfo(
            worker_id=worker.id,
            host=worker.host,
            status=worker.status,
            total_gpus=worker.gpu,
            available_gpus=worker.gpu_available,
            available_gpu_indices=list(worker.gpu_indices_available),
        )


# Global client instance
_client: Optional[PyletClient] = None


async def get_pylet_client(endpoint: Optional[str] = None) -> PyletClient:
    """Get the global Pylet client instance."""
    global _client
    if _client is None:
        _client = PyletClient(endpoint or "http://localhost:8000")
        await _client.init()
    return _client


async def init_pylet_client(endpoint: str) -> PyletClient:
    """Initialize the global Pylet client with a specific endpoint."""
    global _client
    _client = PyletClient(endpoint)
    await _client.init()
    return _client
