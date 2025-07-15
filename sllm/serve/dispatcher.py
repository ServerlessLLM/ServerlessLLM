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
from typing import Any, List, Optional, Dict, Mapping

import aiohttp

from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger
from sllm.serve.model_manager import ModelManager

logger = init_logger(__name__)

DEFAULT_WORKER_API_PORT = 8001
DEFAULT_WORKER_INVOKE_ENDPOINT = "/invoke"
DEFAULT_FORWARD_TIMEOUT = 120  # Timeout for waiting on a response from a worker
DEFAULT_QUEUE_WAIT_TIMEOUT = 10 # Timeout for BRPOP to allow for graceful shutdown checks

class Dispatcher:
    def __init__(self, store: RedisStore, config: Optional[Mapping[str, Any]] = None):
        self.store = store
        self.config = config or {}
        self.is_shutting_down = False

        self.worker_port = self.config.get("worker_api_port", DEFAULT_WORKER_API_PORT)
        self.invoke_endpoint = self.config.get("worker_invoke_endpoint", DEFAULT_WORKER_INVOKE_ENDPOINT)
        self.forward_timeout = self.config.get("forward_timeout", DEFAULT_FORWARD_TIMEOUT)
        self.queue_wait_timeout = self.config.get("queue_wait_timeout", DEFAULT_QUEUE_WAIT_TIMEOUT)

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
            raise RuntimeError("Dispatcher must be started with `start()` before running the consumer loop.")

        logger.info("Starting dispatcher consumer loop...")
        while not self.is_shutting_down:
            try:
                all_models = await self.store.get_all_models()
                if not all_models:
                    logger.debug("No models registered yet. Waiting...")
                    await asyncio.sleep(5)
                    continue

                queue_keys = [ModelManager._get_task_queue_key(m['model_name'], m['backend']) for m in all_models]
                logger.debug(f"Listening on queues: {queue_keys}")

                task = await self.store.dequeue_from_any(queue_keys, timeout=self.queue_wait_timeout)

                if task is None:
                    continue

                queue_name, task_data = task
                asyncio.create_task(self.process_task(queue_name, task_data))

            except Exception as e:
                logger.error(f"An unexpected error occurred in the consumer loop: {e}", exc_info=True)
                await asyncio.sleep(5)


    async def process_task(self, queue_name: str, task_data: Dict[str, Any]) -> None:
        task_id = task_data.get("task_id")
        payload = task_data.get("payload")
        model_identifier = queue_name.replace("queue:", "", 1)
        model_name, backend = model_identifier.split(":", 1)

        if not all([task_id, payload, model_identifier]):
            logger.error(f"Invalid task received from queue '{queue_name}': {task_data}")
            return

        logger.info(f"Processing task {task_id} for model '{model_identifier}'")

        available_instances = await self._find_available_instances(model_identifier)

        if not available_instances:
            logger.warning(f"No available workers for '{model_identifier}'. Requeuing task {task_id} to the front.")
            await self.store.enqueue_task(model_name, backend, task_data)
            return

        target = await self._select_instance_round_robin(model_identifier, available_instances)
        target_worker = target["worker"]
        target_instance_id = target["instance_id"]

        logger.info(f"Dispatching task {task_id} to instance {target_instance_id} on worker {target_worker['node_id']}")

        try:
            worker_response = await self._forward_to_worker(target_worker, target_instance_id, payload)
            await self.store.publish_result(task_id, worker_response)
            logger.info(f"Successfully processed and published result for task {task_id}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Failed to forward task {task_id} to worker {target_worker['node_id']}: {e}. Requeuing.")
            await self.store.enqueue_task(model_name, backend, task_data)
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing task {task_id}: {e}", exc_info=True)
            error_response = {"status": "error", "message": "An internal dispatcher error occurred."}
            await self.store.publish_result(task_id, error_response)



    async def _select_instance_round_robin(self, model_identifier: str, instances: List[Dict]) -> Dict:
        """Selects an instance using a round-robin counter stored in Redis."""
        counter_key = f"rr_counter:{model_identifier}"
        next_index = await self.store.client.incr(counter_key)
        selected_instance = instances[(next_index - 1) % len(instances)]
        return selected_instance

    async def _find_available_instances(self, model_identifier: str) -> List[Dict[str, Any]]:
        """Finds all running instances for a given model from worker heartbeats."""
        all_workers = await self.store.get_all_workers()
        active_instances = []

        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            if model_identifier in instances_on_device:
                for instance_id in instances_on_device[model_identifier]:
                    active_instances.append({"worker": worker, "instance_id": instance_id})
        
        logger.debug(f"Found {len(active_instances)} available instances for {model_identifier}")
        return active_instances

    async def _forward_to_worker(
        self, worker: Dict[str, Any], instance_id: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sends the inference payload to the specified worker instance via HTTP POST."""
        ip_address = worker.get("ip_address")
        if not ip_address:
            raise aiohttp.ClientConnectionError(f"Worker {worker['node_id']} has no IP address in its heartbeat.")

        url = f"http://{ip_address}:{self.worker_port}{self.invoke_endpoint}"
        
        forward_payload = {"instance_id": instance_id, "payload": payload}
        
        async with self.http_session.post(
            url, json=forward_payload, timeout=aiohttp.ClientTimeout(total=self.forward_timeout)
        ) as response:
            response.raise_for_status() 
            return await response.json()
