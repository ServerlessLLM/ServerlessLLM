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
import json
import time
from typing import Any, Dict, List, Mapping, Optional

import aiohttp

from sllm.kv_store import RedisStore
from sllm.logger import init_logger
from sllm.utils import post_json_with_retry

logger = init_logger(__name__)

DEFAULT_WORKER_API_PORT = 8001
DEFAULT_FORWARD_TIMEOUT = 120  # Timeout for waiting on a response from a worker
DEFAULT_QUEUE_WAIT_TIMEOUT = (
    10  # Timeout for BRPOP to allow for graceful shutdown checks
)


class Dispatcher:
    def __init__(
        self, store: RedisStore, config: Optional[Mapping[str, Any]] = None
    ):
        self.store = store
        self.config = config or {}
        self.is_shutting_down = False

        self.worker_port = self.config.get(
            "worker_api_port", DEFAULT_WORKER_API_PORT
        )
        self.forward_timeout = self.config.get(
            "forward_timeout", DEFAULT_FORWARD_TIMEOUT
        )
        self.queue_wait_timeout = self.config.get(
            "queue_wait_timeout", DEFAULT_QUEUE_WAIT_TIMEOUT
        )

        self.http_session: Optional[aiohttp.ClientSession] = None

    def start(self) -> None:
        """Initializes resources for the Dispatcher."""
        logger.info("Starting Dispatcher service...")
        self.http_session = aiohttp.ClientSession()
        self.is_shutting_down = False
        logger.info("Dispatcher service started.")

    async def shutdown(self) -> None:
        """Gracefully shuts down the Dispatcher and its resources."""
        logger.info("Shutting down Dispatcher service...")
        self.is_shutting_down = True
        if self.http_session and not self.http_session.closed:
            await self.http_session.close()
        logger.info("Dispatcher shutdown complete.")

    async def run_consumer_loop(self) -> None:
        """
        The main execution loop for the dispatcher.
        It discovers all model queues and continuously pulls tasks for processing.
        """
        if not self.http_session:
            raise RuntimeError(
                "Dispatcher must be started with `start()` before running the consumer loop."
            )

        logger.info("Starting dispatcher consumer loop...")
        while not self.is_shutting_down:
            try:
                all_models = await self.store.get_all_raw_models()
                if not all_models:
                    logger.debug("No models registered yet. Waiting...")
                    await asyncio.sleep(5)
                    continue

                queue_keys = [
                    f"queue:{m['model']}:{m['backend']}" for m in all_models
                ]
                logger.debug(f"Listening on queues: {queue_keys}")

                task = await self.store.dequeue_from_any(
                    queue_keys, timeout=self.queue_wait_timeout
                )

                if task is None:
                    continue

                queue_name, task_data = task
                asyncio.create_task(self.process_task(queue_name, task_data))

            except Exception as e:
                logger.error(
                    f"An unexpected error occurred in the consumer loop: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(5)

    async def process_task(
        self, queue_name: str, task_data: Dict[str, Any]
    ) -> None:
        task_id = task_data.get("task_id")

        try:
            payload = task_data.get("payload")
            # Handle case where queue_name might be bytes
            if isinstance(queue_name, bytes):
                queue_name = queue_name.decode()
            model_identifier = queue_name.replace("queue:", "", 1)
            model_name, backend = model_identifier.split(":", 1)

            if not all([task_id, payload, model_identifier]):
                logger.error(
                    f"Invalid task received from queue '{queue_name}': {task_data}"
                )
                if task_id:
                    await self._publish_error_result(
                        task_id, "Invalid task data"
                    )
                return

            logger.debug(
                f"Processing task {task_id} for model '{model_identifier}'"
            )

            available_instances = await self._find_available_instances(
                model_identifier
            )

            if not available_instances:
                logger.debug(
                    f"No available workers for '{model_identifier}'. Requesting scaling and requeuing task {task_id}."
                )
                # Atomically increment scaling request to avoid race conditions
                decision_key = f"scaling_decision:{model_name}:{backend}"
                current_decision = await self.store.client.get(decision_key)
                if current_decision is None:
                    # Only set if no scaling decision exists (first request)
                    await self.store.client.set(decision_key, 1, ex=60)
                    logger.info(f"Set scaling decision for {model_identifier}: +1 instance")
                else:
                    logger.debug(f"Scaling already requested for {model_identifier}")
                
                await self.store.enqueue_task(model_name, backend, task_data)
                return

            target = await self._select_instance_round_robin(
                model_identifier, available_instances
            )
            target_worker = target["worker"]
            target_instance_id = target["instance_id"]

            logger.debug(
                f"Dispatching task {task_id} to instance {target_instance_id} on worker {target_worker['node_id']} port {target.get('port')}"
            )

            try:
                # Handle LoRA adapter loading if present in payload
                if "lora_adapter_name" in payload:
                    await self._handle_lora_loading(target, payload)

                worker_response = await self._forward_to_worker(target, payload)
                await self.store.publish_result(task_id, worker_response)
                logger.debug(
                    f"Successfully processed and published result for task {task_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to forward task {task_id} to worker {target_worker['node_id']} after retries: {e}. Requeuing."
                )
                await self.store.enqueue_task(model_name, backend, task_data)

        except Exception as e:
            logger.error(
                f"Critical error processing task {task_id}: {e}", exc_info=True
            )
            if task_id:
                await self._publish_error_result(
                    task_id, f"Internal error: {str(e)}"
                )

    async def _publish_error_result(
        self, task_id: str, error_message: str
    ) -> None:
        """Publish an error result to notify listeners of task failure."""
        error_response = {
            "error": {"message": error_message, "type": "internal_error"}
        }
        try:
            await self.store.publish_result(task_id, error_response)
            logger.info(f"Published error result for task {task_id}")
        except Exception as e:
            logger.error(
                f"Failed to publish error result for task {task_id}: {e}"
            )

    async def _handle_lora_loading(
        self, target: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        """Handle LoRA adapter loading before processing the main request."""
        lora_adapter_name = payload.get("lora_adapter_name")
        if not lora_adapter_name:
            return

        # Get LoRA adapter path from KV store
        model_identifier = f"{payload.get('model_name', '')}:{payload.get('backend', 'transformers')}"
        lora_adapters = await self.store.get_lora_adapters(model_identifier)

        if lora_adapter_name not in lora_adapters:
            raise ValueError(
                f"LoRA adapter '{lora_adapter_name}' not found for model '{model_identifier}'"
            )

        lora_path = lora_adapters[lora_adapter_name]
        worker = target["worker"]
        instance_id = target["instance_id"]
        instance_port = target.get("port")

        node_ip = worker.get("node_ip")
        if not node_ip or not instance_port:
            raise ValueError(f"Invalid worker configuration for LoRA loading")

        # Load LoRA adapter on the target instance
        lora_payload = {
            "instance_id": instance_id,
            "payload": {"lora_name": lora_adapter_name, "lora_path": lora_path},
        }

        url = f"http://{node_ip}:{instance_port}/load_lora_adapter"

        try:
            await post_json_with_retry(
                session=self.http_session,
                url=url,
                payload=lora_payload,
                max_retries=3,
                timeout=self.forward_timeout,
            )
            logger.debug(
                f"Successfully loaded LoRA adapter '{lora_adapter_name}' on instance {instance_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to load LoRA adapter '{lora_adapter_name}': {e}"
            )
            raise

    async def _select_instance_round_robin(
        self, model_identifier: str, instances: List[Dict]
    ) -> Dict:
        """Selects an instance using a round-robin counter stored in Redis."""
        counter_key = f"rr_counter:{model_identifier}"
        try:
            next_index = await self.store._execute_with_retry(
                self.store.client.incr, counter_key
            )
            selected_instance = instances[(next_index - 1) % len(instances)]
            return selected_instance
        except Exception as e:
            logger.warning(
                f"Failed to use Redis counter for round-robin, falling back to local selection: {e}"
            )
            # Fallback to simple modulo selection based on instance count

            fallback_index = int(time.time()) % len(instances)
            return instances[fallback_index]

    async def _find_available_instances(
        self, model_identifier: str
    ) -> List[Dict[str, Any]]:
        """Finds all running instances for a given model from worker heartbeats."""
        all_workers = await self.store.get_all_workers()
        active_instances = []

        for worker in all_workers:
            # Parse instances_on_device from JSON string (stored format in Redis)
            try:
                instances_on_device_str = worker.get(
                    "instances_on_device", "{}"
                )
                if isinstance(instances_on_device_str, str):
                    instances_on_device = json.loads(instances_on_device_str)
                else:
                    instances_on_device = instances_on_device_str
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    f"Failed to parse instances_on_device for worker {worker.get('node_id')}: {e}"
                )
                continue

            if model_identifier in instances_on_device:
                instance_dict = instances_on_device[model_identifier]
                if isinstance(instance_dict, dict):
                    for instance_id, instance_info in instance_dict.items():
                        if instance_info.get("status") == "running":
                            active_instances.append(
                                {
                                    "worker": worker,
                                    "instance_id": instance_id,
                                    "port": instance_info.get("port"),
                                    "endpoint": instance_info.get("endpoint"),
                                }
                            )

        if len(active_instances) == 0:
            logger.debug(f"No available instances found for {model_identifier}")
        else:
            logger.debug(
                f"Found {len(active_instances)} available instances for {model_identifier}"
            )
        return active_instances

    async def _forward_to_worker(
        self, target: Dict[str, Any], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sends the request payload to the specified worker instance via HTTP POST."""
        worker = target["worker"]
        instance_id = target["instance_id"]
        instance_port = target.get("port")

        node_ip = worker.get("node_ip")
        if not node_ip:
            raise aiohttp.ClientConnectionError(
                f"Worker {worker['node_id']} has no IP address in its heartbeat."
            )

        if not instance_port:
            raise aiohttp.ClientConnectionError(
                f"Instance {instance_id} has no port information."
            )

        # Determine request type and endpoint
        request_type = payload.get("action", "generate")  # Default to generate
        if request_type == "fine_tuning":
            endpoint = "/fine-tuning"
        elif request_type == "encode":
            endpoint = "/v1/embeddings"
        else:
            endpoint = "/v1/chat/completions"  # Default inference endpoint

        url = f"http://{node_ip}:{instance_port}{endpoint}"
        logger.info(f"dispatching to URL: {url} with model: {payload.get('model', 'NOT_SET')}")

        # For fine-tuning requests, add concurrency limit
        if request_type == "fine_tuning":
            payload["concurrency"] = 1

        try:
            response = await post_json_with_retry(
                session=self.http_session,
                url=url,
                payload=payload,
                max_retries=3,
                timeout=self.forward_timeout,
            )
            return response
        except Exception as e:
            raise aiohttp.ClientConnectionError(
                f"Failed to forward to worker {worker['node_id']} after retries: {e}"
            )
