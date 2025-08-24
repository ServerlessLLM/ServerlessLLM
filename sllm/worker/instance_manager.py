# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from sllm.logger import init_logger
from sllm.worker.model_downloader import (
    VllmModelDownloader,
    download_lora_adapter,
    download_transformers_model,
)
from sllm.worker.utils import (
    allocate_backend_port,
    validate_lora_adapter_path,
    validate_storage_path,
    validate_transformers_model_path,
    validate_vllm_model_path,
)
from sllm_store.client import SllmStoreClient

logger = init_logger(__name__)


class InstanceManager:
    def __init__(
        self,
        node_ip: str = "127.0.0.1",
        mem_pool_size: int = 1024 * 1024 * 1024 * 8,
        chunk_size: int = 1024 * 1024,
    ):
        self._running_instances: Dict[str, Dict[str, Any]] = {}
        self._instances_lock = asyncio.Lock()  # Protect concurrent access
        self._instance_lookup: Dict[
            str, str
        ] = {}  # instance_id -> model_identifier for fast lookup

        # Store node IP for endpoint creation
        self.node_ip = node_ip

        # SLLM Store Client - based on node IP
        self.client = SllmStoreClient(f"{node_ip}:8073")

        # Track models available on disk: {model_name: (model_paths, model_size)}
        self.disk_models: Dict[str, tuple] = {}

        # Memory management fields (moved from SllmLocalStore)
        self.pinned_memory_pool = {}
        self.chunk_size = chunk_size
        self.pinned_memory_pool_chunks = mem_pool_size // chunk_size
        self.pinned_memory_pool_usage = 0
        self.queued_models = {}
        self.loading_queue = []  # Simplified queue without timing estimates
        self.memory_lock = asyncio.Lock()  # Separate lock for memory operations

        # Scan for existing models on disk
        self.scan_disk_models()

        # Start loading loop
        self.loader = asyncio.create_task(self.loading_loop())

    def scan_disk_models(self) -> None:
        """Scan storage directory and populate disk_models with available models."""
        storage_path = os.getenv("STORAGE_PATH", "./models")
        storage_path = os.path.abspath(storage_path)

        if not os.path.exists(storage_path):
            logger.warning(f"Storage path {storage_path} does not exist")
            return

        # TODO: optimize this loop
        # Scan vLLM models
        vllm_path = os.path.join(storage_path, "vllm")
        if os.path.exists(vllm_path):
            for org_name in os.listdir(vllm_path):
                org_dir = os.path.join(vllm_path, org_name)
                if os.path.isdir(org_dir):
                    for model_name in os.listdir(org_dir):
                        model_dir = os.path.join(org_dir, model_name)
                        if os.path.isdir(model_dir) and validate_vllm_model_path(model_dir):
                            full_model_name = f"{org_name}/{model_name}"
                            model_paths = []
                            total_size = 0
                            rank_count = 0

                            for item in os.listdir(model_dir):
                                if item.startswith("rank_") and os.path.isdir(os.path.join(model_dir, item)):
                                    rank_path = os.path.join(model_dir, item)
                                    model_paths.append(f"vllm/{full_model_name}/{item}")
                                    rank_count += 1
                                    for root, dirs, files in os.walk(rank_path):
                                        total_size += sum(
                                            os.path.getsize(os.path.join(root, file))
                                            for file in files
                                        )

                            if model_paths:
                                self.disk_models[f"{full_model_name}:vllm"] = (
                                    model_paths,
                                    total_size,
                                )

        # Scan transformers models
        transformers_path = os.path.join(storage_path, "transformers")
        if os.path.exists(transformers_path):
            for org_name in os.listdir(transformers_path):
                org_dir = os.path.join(transformers_path, org_name)
                if os.path.isdir(org_dir):
                    for model_name in os.listdir(org_dir):
                        model_dir = os.path.join(org_dir, model_name)
                        if os.path.isdir(model_dir) and validate_transformers_model_path(model_dir):
                            full_model_name = f"{org_name}/{model_name}"
                            model_path = f"transformers/{full_model_name}"
                            total_size = 0
                            for root, dirs, files in os.walk(model_dir):
                                total_size += sum(
                                    os.path.getsize(os.path.join(root, file))
                                    for file in files
                                )
                            self.disk_models[f"{full_model_name}:transformers"] = (
                                [model_path],
                                total_size,
                            )

        vllm_count = len(
            [k for k in self.disk_models.keys() if k.endswith(":vllm")]
        )
        transformers_count = len(
            [k for k in self.disk_models.keys() if k.endswith(":transformers")]
        )
        logger.info(
            f"Found {vllm_count} vLLM, {transformers_count} Transformers models"
        )

    async def _ensure_model_downloaded(
        self, model_config: Dict[str, Any]
    ) -> None:
        """Ensure the model is downloaded before starting the backend."""
        model = model_config.get("model")
        backend = model_config.get("backend")

        if not model or not backend:
            raise ValueError(
                "model_config must contain 'model' and 'backend' keys"
            )
        backend_config = model_config.get("backend_config", {})

        storage_path = os.getenv("STORAGE_PATH", "./models")
        storage_path = os.path.abspath(storage_path)
        if not validate_storage_path(storage_path):
            raise ValueError(
                f"Invalid or inaccessible storage path: {storage_path}"
            )

        if backend == "vllm":
            model_path = os.path.join(storage_path, "vllm", model)
            if not validate_vllm_model_path(model_path):
                if os.path.exists(model_path):
                    logger.warning(
                        f"Incomplete vLLM at {model_path}, re-downloading"
                    )
                    shutil.rmtree(model_path)

                logger.info(f"Downloading vLLM {model}")
                try:
                    downloader = VllmModelDownloader()

                    pretrained_model_name_or_path = backend_config.get(
                        "pretrained_model_name_or_path", model
                    )
                    torch_dtype = backend_config.get("torch_dtype", "float16")
                    tensor_parallel_size = backend_config.get(
                        "tensor_parallel_size", 1
                    )
                    pattern = backend_config.get("pattern", None)
                    max_size = backend_config.get("max_size", None)

                    await downloader.download_vllm_model(
                        model_name=model,
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        tensor_parallel_size=tensor_parallel_size,
                        pattern=pattern,
                        max_size=max_size,
                    )

                    if not validate_vllm_model_path(model_path):
                        raise RuntimeError(
                            f"Model download incomplete: {model_path} validation failed"
                        )

                    logger.debug(f"Downloaded vLLM {model}")
                except Exception as e:
                    logger.error(f"vLLM download failed {model}: {e}")
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    raise
            else:
                logger.info(f"vLLM {model} exists")

        elif backend == "transformers":
            model_path = os.path.join(storage_path, "transformers", model)
            if not validate_transformers_model_path(model_path):
                if os.path.exists(model_path):
                    logger.warning(
                        f"Incomplete transformers at {model_path}, re-downloading"
                    )
                    shutil.rmtree(model_path)

                logger.info(f"Downloading transformers {model}")
                try:
                    pretrained_model_name_or_path = backend_config.get(
                        "pretrained_model_name_or_path", model
                    )
                    torch_dtype = backend_config.get("torch_dtype", "float16")
                    hf_model_class = backend_config.get(
                        "hf_model_class", "AutoModelForCausalLM"
                    )

                    await download_transformers_model(
                        model_name=model,
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        hf_model_class=hf_model_class,
                    )

                    if not validate_transformers_model_path(model_path):
                        raise RuntimeError(
                            f"Model download incomplete: {model_path} validation failed"
                        )

                    logger.info(f"Downloaded transformers {model}")
                except Exception as e:
                    logger.error(f"Transformers download failed {model}: {e}")
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    raise
            else:
                logger.info(f"Transformers {model} exists")

            # Handle LoRA adapter if specified
            adapter_name_or_path = backend_config.get("adapter_name_or_path")
            if adapter_name_or_path:
                adapter_path = os.path.join(
                    storage_path, "transformers", adapter_name_or_path
                )
                if not validate_lora_adapter_path(adapter_path):
                    if os.path.exists(adapter_path):
                        logger.warning(
                            f"Incomplete LoRA at {adapter_path}, re-downloading"
                        )
                        shutil.rmtree(adapter_path)

                    adapter_name = backend_config.get(
                        "adapter_name", adapter_name_or_path
                    )
                    await download_lora_adapter(
                        base_model_name=model,
                        adapter_name=adapter_name,
                        adapter_name_or_path=adapter_name_or_path,
                        hf_model_class=hf_model_class,
                        torch_dtype=torch_dtype,
                    )

                    if not validate_lora_adapter_path(adapter_path):
                        raise RuntimeError(
                            f"LoRA adapter download incomplete: {adapter_path} validation failed"
                        )
                else:
                    logger.info(f"LoRA {adapter_name_or_path} exists")

        elif backend == "dummy":
            # Dummy backend doesn't need model downloading
            pass
        else:
            logger.warning(
                f"Unknown backend {backend}, skipping model download"
            )

    async def _ensure_model_registered(
        self, model_name: str, backend: str, model_config: Dict[str, Any]
    ) -> None:
        """Ensure model is registered with the checkpoint store."""
        try:
            model_identifier = f"{model_name}:{backend}"
            logger.info(f"Registering {model_identifier}")

            if backend == "vllm":
                # Register vLLM model with rank-based paths
                backend_config = model_config.get("backend_config", {})
                tensor_parallel_size = backend_config.get(
                    "tensor_parallel_size", 1
                )

                model_path = f"vllm/{model_name}"
                total_model_size = 0
                model_path_list = []

                logger.info(f"vLLM {tensor_parallel_size} ranks")

                for rank in range(tensor_parallel_size):
                    model_rank_path = f"{model_path}/rank_{rank}"
                    logger.debug(f"Registering {model_rank_path}")
                    model_size = self.client.register_model(model_rank_path)
                    logger.debug(f"Rank {rank}: {model_size} bytes")
                    if model_size > 0:
                        total_model_size += model_size
                        model_path_list.append(model_rank_path)

                # Update disk_models registry
                if model_path_list:
                    self.disk_models[model_identifier] = (
                        model_path_list,
                        total_model_size,
                    )

                logger.info(
                    f"vLLM registered: {model_name}, {len(model_path_list)} ranks, {total_model_size} bytes"
                )

            elif backend == "transformers":
                # Register transformers model
                model_path = f"transformers/{model_name}"
                logger.debug(f"Registering {model_path}")
                model_size = self.client.register_model(model_path)
                logger.debug(f"Registration: {model_size} bytes")
                if model_size > 0:
                    # Update disk_models registry
                    self.disk_models[model_identifier] = (
                        [model_path],
                        model_size,
                    )
                    logger.info(
                        f"Transformers registered: {model_name}, {model_size} bytes"
                    )

        except Exception as e:
            logger.error(f"Registration failed {model_name}: {e}")
            # Don't raise - allow instance to continue, registration might not be critical

    async def start_instance(
        self, model_config: Dict[str, Any], instance_id: Optional[str] = None
    ) -> str:
        model_identifier = f"{model_config['model']}:{model_config['backend']}"
        model = model_config.get("model")
        backend = model_config.get("backend")

        if instance_id is None:
            instance_id = self._generate_instance_id(model, backend)

        logger.debug(f"Starting {instance_id} ({backend})")

        await self._ensure_model_downloaded(model_config)

        # Register model with checkpoint store
        await self._ensure_model_registered(model, backend, model_config)

        # Load model to pinned memory pool (hot potato mechanism)
        model_identifier = f"{model}:{backend}"
        await self.load_to_host(model_identifier)

        backend_config = model_config.get("backend_config", {})
        startup_config = model_config.get("startup_config", {})

        # Allocate port for this instance
        allocated_port = allocate_backend_port(backend)
        backend_config["port"] = allocated_port
        logger.info(f"Port {allocated_port}: {instance_id}")

        if backend == "vllm":
            from sllm.backends.vllm_backend import VllmBackend

            model_backend_cls = VllmBackend
        elif backend == "dummy":
            from sllm.backends.dummy_backend import DummyBackend

            model_backend_cls = DummyBackend
        elif backend == "transformers":
            from sllm.backends.transformers_backend import (
                TransformersBackend,
            )

            model_backend_cls = TransformersBackend
        else:
            logger.error(f"Unknown backend: {backend}")
            raise ValueError(f"Unknown backend: {backend}")

        # Create and initialize the backend instance
        backend_instance = None
        try:
            backend_instance = model_backend_cls(model, backend_config)

            # Initialize the backend
            await backend_instance.init_backend()

            # Store the instance with proper synchronization
            async with self._instances_lock:
                if model_identifier not in self._running_instances:
                    self._running_instances[model_identifier] = {}

                instance_url = f"http://{self.node_ip}:{allocated_port}"
                logger.debug(f"Started {instance_id}: {instance_url}")

                self._running_instances[model_identifier][instance_id] = {
                    "backend": backend_instance,
                    "model_config": model_config,
                    "status": "RUNNING",
                    "host": backend_instance.host,
                    "port": allocated_port,
                    "endpoint": f"{self.node_ip}:{allocated_port}",
                }

                # Update reverse lookup for performance
                self._instance_lookup[instance_id] = model_identifier

            logger.debug(f"Started {instance_id}")

        except Exception as e:
            logger.error(f"Start failed {instance_id}: {e}")
            # Cleanup on failure
            if backend_instance:
                try:
                    await backend_instance.shutdown()
                except Exception as cleanup_e:
                    logger.warning(
                        f"Failed to cleanup backend during error recovery: {cleanup_e}"
                    )
            raise

        return instance_id

    async def stop_instance(self, instance_id: str) -> bool:
        logger.debug(f"Stopping {instance_id}")

        async with self._instances_lock:
            # Use reverse lookup for performance
            model_identifier = self._instance_lookup.get(instance_id)
            if not model_identifier:
                logger.warning(f"{instance_id} not found")
                return False

            instances = self._running_instances.get(model_identifier)
            if not instances or instance_id not in instances:
                logger.warning(f"{instance_id} not in running instances")
                # Clean up orphaned lookup entry
                if instance_id in self._instance_lookup:
                    del self._instance_lookup[instance_id]
                return False

            try:
                instance_info = instances[instance_id]
                backend = instance_info["backend"]

                # Shutdown backend (do this outside the lock to avoid holding lock during slow operation)
                # But mark as stopping first
                instance_info["status"] = "STOPPING"
            except Exception as e:
                logger.error(f"Stop prep failed {instance_id}: {e}")
                return False

        # Shutdown backend outside of lock
        try:
            await backend.shutdown()
        except Exception as e:
            logger.error(f"Shutdown failed {instance_id}: {e}")
            # Continue with cleanup even if shutdown failed

        # Remove from data structures
        async with self._instances_lock:
            try:
                del instances[instance_id]
                del self._instance_lookup[instance_id]

                # Clean up empty model identifier entries
                if not instances:
                    del self._running_instances[model_identifier]
                    # Unload from pinned memory if no instances are using this model
                    await self._unload_from_host(model_identifier)

                logger.debug(f"Stopped {instance_id}")
                return True
            except KeyError:
                logger.warning(f"{instance_id} already removed")
                return True

    async def run_inference(
        self, instance_id: str, request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Use reverse lookup for performance - only need read lock
        async with self._instances_lock:
            model_identifier = self._instance_lookup.get(instance_id)
            if not model_identifier:
                logger.warning(f"{instance_id} not found for inference")
                return {"error": f"Instance {instance_id} not found"}

            instances = self._running_instances.get(model_identifier)
            if not instances or instance_id not in instances:
                logger.warning(f"{instance_id} not in running instances")
                return {"error": f"Instance {instance_id} not found"}

            instance_info = instances[instance_id]
            if instance_info.get("status") != "RUNNING":
                logger.warning(
                    f"{instance_id} state: {instance_info.get('status')}"
                )
                return {"error": f"Instance {instance_id} is not available"}

            backend = instance_info["backend"]

        # Run inference outside of lock to allow concurrent requests
        try:
            if (
                request_payload.get("task") == "embedding"
                or "input" in request_payload
            ):
                result = await backend.encode(request_payload)
            else:
                result = await backend.generate(request_payload)

            return result

        except Exception as e:
            logger.error(f"Inference failed {instance_id}: {e}")
            return {"error": f"Inference failed: {str(e)}"}

    def get_running_instances_info(self) -> Dict[str, Any]:
        # This is called from heartbeat, so make it thread-safe
        info = {}
        try:
            # Take a snapshot to avoid holding lock too long
            instances_snapshot = dict(self._running_instances)
            for model_identifier, instances in instances_snapshot.items():
                info[model_identifier] = {}
                for instance_id, instance_data in instances.items():
                    info[model_identifier][instance_id] = {
                        "status": instance_data.get("status", "UNKNOWN"),
                        "host": instance_data.get("host", "0.0.0.0"),
                        "port": instance_data.get("port", 0),
                        "endpoint": instance_data.get(
                            "endpoint", f"{self.node_ip}:0"
                        ),
                    }
        except Exception as e:
            logger.warning(f"Error getting instances info: {e}")
            info = {}
        return info

    def _generate_instance_id(self, model: str, backend: str) -> str:
        unique_part = uuid.uuid4().hex[:8]
        return f"{model}-{backend}-{unique_part}"

    # Memory management functions (moved and simplified from SllmLocalStore)
    async def load_to_host(self, model_name: str) -> bool:
        """Load model to pinned memory pool (hot potato mechanism)"""
        async with self.memory_lock:
            if model_name not in self.disk_models:
                logger.error(f"{model_name} not found on node {self.node_ip}")
                return False
            if model_name in self.pinned_memory_pool:
                logger.info(f"{model_name} already loaded")
                return True
            elif model_name in self.queued_models:
                logger.info(f"{model_name} loading")
                return True

            # Add to loading queue
            self.loading_queue.append(model_name)
            self.queued_models[model_name] = True
            logger.info(f"{model_name} queued")
            return True

    async def loading_loop(self):
        """Simplified loading loop without timing estimates"""
        while True:
            async with self.memory_lock:
                if len(self.loading_queue) == 0:
                    await asyncio.sleep(1)
                    continue
                model_name = self.loading_queue[0]
                logger.info(f"Loading {model_name}")

            if model_name not in self.disk_models:
                logger.error(f"Model {model_name} not found in disk_models")
                async with self.memory_lock:
                    self.loading_queue.pop(0)
                    self.queued_models.pop(model_name, None)
                continue

            model_path_list, model_size = self.disk_models[model_name]
            can_load = await self._lru_eviction(model_size)
            if not can_load:
                logger.warning(f"{model_name} cannot be loaded: insufficient memory (need {model_size} bytes)")
                await asyncio.sleep(1)
                continue

            # Attempt to load all model paths
            logger.debug(f"Loading {model_name}")
            success = True
            for model_path in model_path_list:
                try:
                    if not self.client.load_into_cpu(model_path):
                        logger.error(f"Failed to load model path: {model_path}")
                        success = False
                        break
                except Exception as e:
                    logger.error(f"Exception loading model path {model_path}: {e}")
                    success = False
                    break

            async with self.memory_lock:
                self.loading_queue.pop(0)
                self.queued_models.pop(model_name, None)

                if success:
                    self.pinned_memory_pool[model_name] = time.time()
                    self.pinned_memory_pool_usage += (
                        model_size + self.chunk_size - 1
                    ) // self.chunk_size
                    logger.info(f"{model_name} loaded")
                else:
                    logger.error(f"Failed to load {model_name}: model loading unsuccessful")

    async def _lru_eviction(self, model_size):
        """Evict least recently used models to make space"""
        async with self.memory_lock:
            sorted_models = sorted(
                self.pinned_memory_pool.items(), key=lambda x: x[1]
            )
            required_chunks = (
                model_size + self.chunk_size - 1
            ) // self.chunk_size

            logger.debug(
                f"Memory: {self.pinned_memory_pool_usage}/{self.pinned_memory_pool_chunks}, need {required_chunks}"
            )

            while (
                self.pinned_memory_pool_usage + required_chunks
                > self.pinned_memory_pool_chunks
                and len(sorted_models) > 0
            ):
                model_name, _ = sorted_models.pop(0)
                if model_name not in self.queued_models:
                    model_path_list, _ = self.disk_models[model_name]
                    for model_path in model_path_list:
                        self.client.unload_from_cpu(model_path)
                    self.pinned_memory_pool.pop(model_name)
                    unloaded_chunks = (
                        self.disk_models[model_name][1] + self.chunk_size - 1
                    ) // self.chunk_size
                    self.pinned_memory_pool_usage -= unloaded_chunks
                    logger.debug(
                        f"{model_name} evicted {unloaded_chunks} chunks"
                    )

            return (
                self.pinned_memory_pool_usage + required_chunks
                <= self.pinned_memory_pool_chunks
            )

    async def _unload_from_host(self, model_name: str):
        """Unload model from pinned memory pool"""
        async with self.memory_lock:
            if model_name not in self.pinned_memory_pool:
                return

            if model_name not in self.disk_models:
                return

            model_path_list, model_size = self.disk_models[model_name]
            for model_path in model_path_list:
                self.client.unload_from_cpu(model_path)

            self.pinned_memory_pool.pop(model_name)
            unloaded_chunks = (
                model_size + self.chunk_size - 1
            ) // self.chunk_size
            self.pinned_memory_pool_usage -= unloaded_chunks
            logger.debug(f"{model_name} unloaded ({unloaded_chunks} chunks)")

    async def register_lora_adapter(
        self,
        base_model_name: str,
        adapter_name: str,
        adapter_path: str,
        backend_config: Dict[str, Any],
    ) -> int:
        """Register and download LoRA adapter for a base model"""
        base_model_identifier = f"{base_model_name}:transformers"
        if base_model_identifier not in self.disk_models:
            logger.error(f"Base model {base_model_name} not found on this node")
            return -1

        # Check if LoRA adapter already exists
        storage_path = os.getenv("STORAGE_PATH", "./models")
        storage_path = os.path.abspath(storage_path)
        local_adapter_path = os.path.join(
            storage_path,
            "transformers",
            base_model_name,
            "adapters",
            adapter_name,
        )

        if os.path.exists(local_adapter_path) and validate_lora_adapter_path(
            local_adapter_path
        ):
            logger.info(f"LoRA {adapter_name} exists")
            return 0  # Success, already exists

        hf_model_class = backend_config.get(
            "hf_model_class", "AutoModelForCausalLM"
        )
        torch_dtype = backend_config.get("torch_dtype", "float16")

        logger.info(f"Downloading LoRA {adapter_path} for {base_model_name}")

        try:
            return await download_lora_adapter(
                base_model_name,
                adapter_name,
                adapter_path,
                hf_model_class,
                torch_dtype,
            )
        except Exception as e:
            logger.error(f"Failed to download LoRA adapter {adapter_name}: {e}")
            return -1
