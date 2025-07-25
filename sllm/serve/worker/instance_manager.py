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
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from sllm.serve.logger import init_logger
from sllm.serve.worker.model_downloader import (
    VllmModelDownloader,
    download_lora_adapter,
    download_transformers_model,
)
from sllm.serve.worker.utils import (
    validate_storage_path,
    validate_vllm_model_path,
    validate_transformers_model_path,
    validate_lora_adapter_path,
)

logger = init_logger(__name__)



class InstanceManager:
    def __init__(self):
        self._running_instances: Dict[str, Dict[str, Any]] = {}
        self._instances_lock = asyncio.Lock()  # Protect concurrent access
        self._instance_lookup: Dict[
            str, str
        ] = {}  # instance_id -> model_identifier for fast lookup

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
                    logger.warning(f"Incomplete vLLM model found at {model_path}, re-downloading")
                    shutil.rmtree(model_path)
                
                logger.info(f"Downloading VLLM model {model} to {model_path}")
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

                    logger.info(f"Successfully downloaded VLLM model {model}")
                except Exception as e:
                    logger.error(f"Failed to download VLLM model {model}: {e}")
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    raise
            else:
                logger.info(f"VLLM model {model} already exists and is valid")

        elif backend == "transformers":
            model_path = os.path.join(storage_path, "transformers", model)
            if not validate_transformers_model_path(model_path):
                if os.path.exists(model_path):
                    logger.warning(f"Incomplete transformers model found at {model_path}, re-downloading")
                    shutil.rmtree(model_path)
                
                logger.info(f"Downloading Transformers model {model} to {model_path}")
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

                    logger.info(f"Successfully downloaded Transformers model {model}")
                except Exception as e:
                    logger.error(f"Failed to download Transformers model {model}: {e}")
                    if os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    raise
            else:
                logger.info(f"Transformers model {model} already exists and is valid")

            # Handle LoRA adapter if specified
            adapter_name_or_path = backend_config.get("adapter_name_or_path")
            if adapter_name_or_path:
                adapter_path = os.path.join(storage_path, "transformers", adapter_name_or_path)
                if not validate_lora_adapter_path(adapter_path):
                    if os.path.exists(adapter_path):
                        logger.warning(f"Incomplete LoRA adapter found at {adapter_path}, re-downloading")
                        shutil.rmtree(adapter_path)
                    
                    adapter_name = backend_config.get("adapter_name", adapter_name_or_path)
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
                    logger.info(f"LoRA adapter {adapter_name_or_path} already exists and is valid")

        elif backend == "dummy":
            # Dummy backend doesn't need model downloading
            pass
        else:
            logger.warning(
                f"Unknown backend {backend}, skipping model download"
            )

    async def start_instance(
        self, model_config: Dict[str, Any], instance_id: Optional[str] = None
    ) -> str:
        model_identifier = f"{model_config['model']}:{model_config['backend']}"
        model = model_config.get("model")
        backend = model_config.get("backend")

        if instance_id is None:
            instance_id = self._generate_instance_id(model, backend)

        logger.info(f"Starting instance {instance_id} with backend {backend}")

        await self._ensure_model_downloaded(model_config)
        backend_config = model_config.get("backend_config", {})
        startup_config = model_config.get("startup_config", {})

        if backend == "vllm":
            from sllm.serve.backends.vllm_backend import VllmBackend

            model_backend_cls = VllmBackend
        elif backend == "dummy":
            from sllm.serve.backends.dummy_backend import DummyBackend

            model_backend_cls = DummyBackend
        elif backend == "transformers":
            from sllm.serve.backends.transformers_backend import (
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

                self._running_instances[model_identifier][instance_id] = {
                    "backend": backend_instance,
                    "model_config": model_config,
                    "status": "running",
                }

                # Update reverse lookup for performance
                self._instance_lookup[instance_id] = model_identifier

            logger.info(f"Successfully started instance {instance_id}")

        except Exception as e:
            logger.error(f"Failed to start instance {instance_id}: {e}")
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
        logger.info(f"Stopping instance {instance_id}")

        async with self._instances_lock:
            # Use reverse lookup for performance
            model_identifier = self._instance_lookup.get(instance_id)
            if not model_identifier:
                logger.warning(f"Instance {instance_id} not found")
                return False

            instances = self._running_instances.get(model_identifier)
            if not instances or instance_id not in instances:
                logger.warning(
                    f"Instance {instance_id} not found in running instances"
                )
                # Clean up orphaned lookup entry
                if instance_id in self._instance_lookup:
                    del self._instance_lookup[instance_id]
                return False

            try:
                instance_info = instances[instance_id]
                backend = instance_info["backend"]

                # Shutdown backend (do this outside the lock to avoid holding lock during slow operation)
                # But mark as stopping first
                instance_info["status"] = "stopping"
            except Exception as e:
                logger.error(
                    f"Failed to prepare instance {instance_id} for stopping: {e}"
                )
                return False

        # Shutdown backend outside of lock
        try:
            await backend.shutdown()
        except Exception as e:
            logger.error(
                f"Failed to shutdown backend for instance {instance_id}: {e}"
            )
            # Continue with cleanup even if shutdown failed

        # Remove from data structures
        async with self._instances_lock:
            try:
                del instances[instance_id]
                del self._instance_lookup[instance_id]

                # Clean up empty model identifier entries
                if not instances:
                    del self._running_instances[model_identifier]

                logger.info(f"Successfully stopped instance {instance_id}")
                return True
            except KeyError:
                logger.warning(
                    f"Instance {instance_id} was already removed during shutdown"
                )
                return True

    async def run_inference(
        self, instance_id: str, request_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Use reverse lookup for performance - only need read lock
        async with self._instances_lock:
            model_identifier = self._instance_lookup.get(instance_id)
            if not model_identifier:
                logger.warning(
                    f"Instance {instance_id} not found for inference"
                )
                return {"error": f"Instance {instance_id} not found"}

            instances = self._running_instances.get(model_identifier)
            if not instances or instance_id not in instances:
                logger.warning(
                    f"Instance {instance_id} not found in running instances"
                )
                return {"error": f"Instance {instance_id} not found"}

            instance_info = instances[instance_id]
            if instance_info.get("status") != "running":
                logger.warning(
                    f"Instance {instance_id} is not in running state: {instance_info.get('status')}"
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
            logger.error(
                f"Failed to run inference on instance {instance_id}: {e}"
            )
            return {"error": f"Inference failed: {str(e)}"}

    def get_running_instances_info(self) -> Dict[str, Any]:
        # This is called from heartbeat, so make it thread-safe
        info = {}
        try:
            # Take a snapshot to avoid holding lock too long
            instances_snapshot = dict(self._running_instances)
            for model_identifier, instances in instances_snapshot.items():
                info[model_identifier] = list(instances.keys())
        except Exception as e:
            logger.warning(f"Error getting instances info: {e}")
            info = {}
        return info

    def _generate_instance_id(self, model: str, backend: str) -> str:
        unique_part = uuid.uuid4().hex[:8]
        return f"{model}-{backend}-{unique_part}"
