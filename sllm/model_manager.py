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
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from sllm.kv_store import RedisStore
from sllm.logger import init_logger

logger = init_logger(__name__)
ModelConfig = Dict[str, Any]


class ModelManager:
    def __init__(self, store: RedisStore):
        self.store = store
        self._shutdown_event = asyncio.Event()
        self._background_tasks = []

    def start(self) -> None:
        self._background_tasks.append(
            asyncio.create_task(self._deletion_listener_loop())
        )
        logger.info("ModelManager started with deletion listener loop.")

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        if self._background_tasks:
            await asyncio.gather(
                *self._background_tasks, return_exceptions=True
            )
        logger.info("ModelManager shutdown complete.")

    async def _deletion_listener_loop(self):
        async with self.store.client.pubsub() as pubsub:
            await pubsub.subscribe("model:delete:notifications")
            logger.info("Subscribed to model deletion notifications.")
            active_cleanups = set()

            while not self._shutdown_event.is_set():
                try:
                    message = await pubsub.get_message(
                        ignore_subscribe_messages=True, timeout=1
                    )
                    if message and message.get("type") == "message":
                        data = json.loads(message["data"])
                        model_key = data.get("model_key")

                        if model_key and model_key not in active_cleanups:
                            logger.info(
                                f"Received deletion notification for {model_key}"
                            )
                            active_cleanups.add(model_key)
                            cleanup_task = asyncio.create_task(
                                self._orchestrate_cleanup(model_key)
                            )
                            cleanup_task.add_done_callback(
                                lambda t,
                                key=model_key: active_cleanups.discard(key)
                            )
                        elif model_key in active_cleanups:
                            logger.debug(
                                f"Cleanup already in progress for {model_key}, ignoring duplicate notification"
                            )

                except asyncio.CancelledError:
                    logger.info("Deletion listener loop cancelled.")
                    break
                except Exception as e:
                    logger.error(
                        f"Error in deletion listener loop: {e}", exc_info=True
                    )
                    await asyncio.sleep(5)

    async def register(self, model_config: ModelConfig) -> None:
        model_name = model_config.get("model")
        backend = model_config.get("backend", None)
        if not model_name or not backend:
            raise ValueError(
                "Model configuration must include 'model' and 'backend' keys."
            )

        model_key = self._get_model_key(model_name, backend)
        if await self.store.client.exists(model_key):
            raise ValueError(
                f"Model '{model_name}:{backend}' is already registered."
            )

        logger.info(f"Registering model '{model_name}:{backend}'")
        await self.store.register_model(model_config)
        logger.info(f"Successfully registered model '{model_name}:{backend}'")

    async def update(
        self, model_name: str, backend: str, model_config: ModelConfig
    ) -> Optional[ModelConfig]:
        logger.info(f"Attempting to update model '{model_name}:{backend}'")
        existing_model = await self.store.get_model(model_name, backend)
        if not existing_model:
            logger.warning(
                f"Update failed: model '{model_name}:{backend}' not found."
            )
            return None

        model_config["model"] = model_name
        model_config["backend"] = backend

        await self.store.register_model(model_config)
        logger.info(f"Successfully updated model '{model_name}:{backend}'")
        return model_config

    async def delete(
        self,
        model_name: str,
        backend: Optional[Union[str, List[str]]] = None,
        lora_adapters: Optional[List[str]] = None,
    ) -> bool:
        if lora_adapters is not None:
            backends_to_update: List[str]
            if backend is None:
                logger.info(
                    f"No backend specified for LoRA deletion. Defaulting to 'transformers'."
                )
                backends_to_update = ["transformers"]
            elif isinstance(backend, str):
                if backend == "all":
                    backends_to_update = await self.get_all_backends(model_name)
                else:
                    backends_to_update = [backend]
            else:
                backends_to_update = backend

            for b_name in backends_to_update:
                await self.store.delete_lora_adapters(
                    model_name, b_name, lora_adapters
                )
                logger.info(
                    f"Successfully deleted lora adapters for '{model_name}:{b_name}'"
                )

        backends_to_delete: List[str]
        if backend is None or backend == "all":
            backends_to_delete = await self.get_all_backends(model_name)
        elif isinstance(backend, str):
            backends_to_delete = [backend]
        else:
            backends_to_delete = backend

        if not backends_to_delete:
            logger.warning(
                f"No backends found for model '{model_name}'. Nothing to delete."
            )
            return True

        acquired_locks = []
        try:
            for b_name in backends_to_delete:
                if await self.store.acquire_deletion_lock(model_name, b_name):
                    acquired_locks.append(b_name)
                    logger.info(
                        f"Acquired deletion lock for '{model_name}:{b_name}'"
                    )
                else:
                    logger.warning(
                        f"Could not acquire deletion lock for '{model_name}:{b_name}' - deletion already in progress"
                    )

            for b_name in acquired_locks:
                model_key = self._get_model_key(model_name, b_name)
                await self.store.delete_model(model_key)
                logger.info(
                    f"Initiated deletion for '{model_key}'. Status set to 'excommunicado'."
                )

        finally:
            for b_name in backends_to_delete:
                if b_name not in acquired_locks:
                    continue
                await self.store.release_deletion_lock(model_name, b_name)

        return len(acquired_locks) > 0

    async def _orchestrate_cleanup(self, model_key: str):
        model_name, backend = self._parse_model_key(model_key)

        lock_key = f"deletion_lock:{model_name}:{backend}"
        if not await self.store.client.exists(lock_key):
            logger.warning(
                f"No deletion lock found for {model_key}. Skipping cleanup."
            )
            return

        try:
            task_queue_key = self._get_task_queue_key(model_name, backend)
            model_identifier = f"{model_name}:{backend}"

            logger.info(f"Waiting for queue {task_queue_key} to drain...")
            queue_drain_timeout = 300
            start_time = time.time()

            while await self.get_queue_depth(model_name, backend) > 0:
                if time.time() - start_time > queue_drain_timeout:
                    logger.error(
                        f"Queue drain timeout for {model_key}. Force proceeding with cleanup."
                    )
                    break
                logger.debug(
                    f"Queue depth: {await self.get_queue_depth(model_name, backend)}"
                )
                await asyncio.sleep(5)
            logger.info(f"Queue for {model_key} is empty.")

            logger.info(
                f"Waiting for running instances of {model_identifier} to terminate..."
            )
            instance_drain_timeout = 180
            start_time = time.time()

            while await self._count_running_instances(model_identifier) > 0:
                if time.time() - start_time > instance_drain_timeout:
                    logger.error(
                        f"Instance termination timeout for {model_key}. Force proceeding with cleanup."
                    )
                    break
                logger.debug(
                    f"Running instances: {await self._count_running_instances(model_identifier)}"
                )
                await asyncio.sleep(5)

            logger.info(f"All instances for {model_key} are terminated.")

            # Use atomic model deletion to prevent race conditions
            with self.store._deletion_locks_lock:
                lock_value = self.store._deletion_locks.get(
                    f"{model_name}:{backend}"
                )

            if lock_value:
                (
                    deletion_success,
                    error_msg,
                ) = await self.store.atomic_model_deletion(
                    model_key, "excommunicado", lock_value
                )

                if deletion_success:
                    # Clean up additional resources
                    pipe = self.store.client.pipeline()
                    pipe.delete(task_queue_key)
                    pipe.delete(f"workers:ready:{model_name}:{backend}")
                    pipe.delete(f"workers:busy:{model_name}:{backend}")
                    await pipe.execute()

                    logger.info(
                        f"[{model_key}] Cleanup complete. Model atomically deleted."
                    )
                else:
                    logger.error(
                        f"Atomic model deletion failed for {model_key}: {error_msg}"
                    )

            else:
                logger.warning(
                    f"No deletion lock found for {model_key}, cannot perform atomic deletion"
                )

                # Fallback to old method
                pipe = self.store.client.pipeline()
                pipe.delete(model_key)
                pipe.delete(task_queue_key)
                pipe.delete(f"workers:ready:{model_name}:{backend}")
                pipe.delete(f"workers:busy:{model_name}:{backend}")
                await pipe.execute()

                logger.info(
                    f"[{model_key}] Cleanup complete (fallback method used)."
                )

        except Exception as e:
            logger.error(
                f"Error during cleanup for {model_key}: {e}", exc_info=True
            )

        finally:
            # Always release the deletion lock
            await self.store.release_deletion_lock(model_name, backend)
            logger.info(f"Released deletion lock for {model_key}")

    @staticmethod
    def get_queue_name(model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    async def get_queue_length(self, model_name: str, backend: str) -> int:
        queue_name = self.get_queue_name(model_name, backend)
        length = await self.store.get_queue_length(queue_name)
        return length

    async def get_model(
        self, model_name: str, backend: str
    ) -> Optional[ModelConfig]:
        model = await self.store.get_model(model_name, backend)
        if not model or model.get("status") == "excommunicado":
            return None
        return model

    async def get_all_models(self) -> List[ModelConfig]:
        return await self.store.get_all_raw_models()

    async def get_all_backends(self, model_name: str) -> List[str]:
        all_models = await self.get_all_models()
        backends = [
            model["backend"]
            for model in all_models
            if model.get("model") == model_name
        ]
        return backends

    ### HELPER FUNCTIONS ###
    def _get_model_key(self, model_name: str, backend: str) -> str:
        return f"model:{model_name}:{backend}"

    def _parse_model_key(self, model_key: str) -> Tuple[str, str]:
        """
        Parses a model_key back into its constituent parts.
        Assumes format "model:name-part-1:backend" or "model:org/name:backend"
        """
        try:
            parts = model_key.split(":")
            prefix = parts[0]
            backend = parts[-1]
            model_name = ":".join(parts[1:-1])

            if prefix != "model" or not model_name or not backend:
                raise ValueError("Invalid model key format")

            return model_name, backend
        except (IndexError, ValueError):
            logger.error(f"Could not parse invalid model key: '{model_key}'")
            raise ValueError(f"Invalid model key format: {model_key}")

    def _get_task_queue_key(self, model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    def _get_worker_set_key(
        self, model_name: str, backend: str, state: str
    ) -> str:
        return f"workers:{model_name}:{backend}:{state}"

    async def count_running_instances(self, model_identifier: str) -> int:
        all_workers = await self.store.get_all_workers()
        count = 0
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            count += len(instances_on_device.get(model_identifier, []))
        return count
