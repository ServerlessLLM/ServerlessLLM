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
import random
import time
import uuid
from typing import Any, List, Mapping, Optional, Dict

import aiohttp

from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger

logger = init_logger(__name__)

DEFAULT_WORKER_TIMEOUT = 60
DEFAULT_PRUNE_INTERVAL = 15
DEFAULT_SCALING_LOOP_INTERVAL = 5

class WorkerManager:
    def __init__(self, store: RedisStore, config: Optional[Mapping[str, Any]] = None):
        self.store = store
        self.config = config or {}
        
        self.worker_timeout = self.config.get("worker_timeout", DEFAULT_WORKER_TIMEOUT)
        self.prune_interval = self.config.get("prune_interval", DEFAULT_PRUNE_INTERVAL)
        self.scaling_loop_interval = self.config.get("scaling_loop_interval", DEFAULT_SCALING_LOOP_INTERVAL)
        
        self.http_session: Optional[aiohttp.ClientSession] = None
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []

    def start(self) -> None:
        logger.info("Starting WorkerManager background tasks...")
        self.http_session = aiohttp.ClientSession()
        self._background_tasks.append(asyncio.create_task(self._prune_loop()))
        self._background_tasks.append(asyncio.create_task(self._scaling_loop()))
        logger.info("WorkerManager started with prune and scaling loops.")

    async def shutdown(self) -> None:
        logger.info("Shutting down WorkerManager background tasks...")
        self._shutdown_event.set()
        if self.http_session:
            await self.http_session.close()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info("WorkerManager shutdown complete.")

    async def _scaling_loop(self):
        while not self._shutdown_event.is_set():
            logger.debug("Scaling loop running...")
            try:
                decision_keys = [key.decode('utf-8') async for key in self.store.client.scan_iter("scaling_decision:*")]

                for key in decision_keys:
                    instances_needed_str = await self.store.client.get(key)
                    if not instances_needed_str:
                        continue
                    
                    instances_needed = int(instances_needed_str)
                    _, model_name, backend = key.split(":")

                    logger.info(f"Found scaling task for '{model_name}:{backend}': need {instances_needed} instances.")

                    if instances_needed > 0:
                        await self._execute_scale_up(model_name, backend, instances_needed)
                    elif instances_needed < 0:
                        await self._execute_scale_down(model_name, backend, abs(instances_needed))
                    
                    await self.store.client.delete(key)

                await asyncio.sleep(self.scaling_loop_interval)

            except asyncio.CancelledError:
                logger.info("Scaling loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}", exc_info=True)
                await asyncio.sleep(self.scaling_loop_interval)

    async def _execute_scale_up(self, model_name: str, backend: str, count: int):
        logger.info(f"Executing scale-up for '{model_name}:{backend}' by {count} instance(s).")
        model_config = await self.store.get_model(model_name, backend)
        if not model_config:
            logger.error(f"Cannot scale up: Model config for '{model_name}:{backend}' not found.")
            return

        all_workers = await self.get_all_worker_info()
        model_identifier = f"{model_name}:{backend}"
        
        # TODO: Add more advanced scheduling (e.g., least-loaded by GPU memory)
        eligible_workers = [
            w for w in all_workers 
            if model_identifier in w.get("registered_models", [])
        ]

        if not eligible_workers:
            logger.warning(f"No eligible workers available to scale up for {model_identifier}.")
            return

        for i in range(count):
            target_worker = random.choice(eligible_workers)
            instance_id = self._generate_instance_id(model_name, backend)
            
            logger.info(f"Attempting to start instance {instance_id} on worker {target_worker['node_id']}")
            success = await self._send_start_instance_request(target_worker, instance_id, model_config)
            if not success:
                logger.error(f"Failed to start instance on worker {target_worker['node_id']}. Trying another worker if available.")
    
    async def _execute_scale_down(self, model_name: str, backend: str, count: int):
        logger.info(f"Executing scale-down for '{model_name}:{backend}' by {count} instance(s).")
        
        all_workers = await self.get_all_worker_info()
        model_identifier = f"{model_name}:{backend}"
        
        running_instances = []
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            if model_identifier in instances_on_device:
                for instance_id in instances_on_device[model_identifier]:
                    running_instances.append({"worker": worker, "instance_id": instance_id})

        if not running_instances:
            logger.warning(f"No running instances of {model_identifier} found to scale down.")
            return

        instances_to_stop = random.sample(running_instances, min(count, len(running_instances)))

        for item in instances_to_stop:
            worker = item["worker"]
            instance_id = item["instance_id"]
            logger.info(f"Attempting to stop instance {instance_id} on worker {worker['node_id']}")
            await self._send_stop_instance_request(worker, instance_id)

    async def count_running_instances(self, model_identifier: str) -> int:
        """Counts the total number of active instances for a given model identifier."""
        all_workers = await self.store.get_all_workers()
        count = 0
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            count += len(instances_on_device.get(model_identifier, []))
        return count

    async def _send_start_instance_request(self, worker: Dict[str, Any], instance_id: str, model_config: Dict[str, Any]) -> bool:
        ip_address = worker.get("ip_address")
        if not ip_address:
            logger.error(f"Cannot send command to worker {worker['node_id']}: missing IP address.")
            return False

        url = f"http://{ip_address}:8001/start_instance"
        payload = {
            "instance_id": instance_id,
            "model_config": model_config.get("backend_config", {})
        }
        try:
            async with self.http_session.post(url, json=payload, timeout=30) as response:
                response.raise_for_status()
                logger.info(f"Successfully sent start command for {instance_id} to {worker['node_id']}.")
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"HTTP request to start instance on {worker['node_id']} failed: {e}")
            return False

    async def _send_stop_instance_request(self, worker: Dict[str, Any], instance_id: str) -> bool:
        ip_address = worker.get("ip_address")
        if not ip_address:
            logger.error(f"Cannot send command to worker {worker['node_id']}: missing IP address.")
            return False

        url = f"http://{ip_address}:8001/stop_instance"
        payload = {"instance_id": instance_id}
        try:
            async with self.http_session.post(url, json=payload, timeout=30) as response:
                response.raise_for_status()
                logger.info(f"Successfully sent stop command for {instance_id} to {worker['node_id']}.")
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"HTTP request to stop instance on {worker['node_id']} failed: {e}")
            return False

    async def process_heartbeat(self, payload: Dict[str, Any]) -> None:
        node_id = payload.get("node_id")
        if not node_id:
            logger.warning("Received a heartbeat with no node_id. Ignoring.")
            return

        worker_key = self.store._get_worker_key(node_id)
        redis_hash = {
            "node_id": node_id,
            "ip_address": payload.get("ip_address"),
            "registered_models": json.dumps(payload.get("registered_models", [])),
            "hardware_info": json.dumps(payload.get("hardware_info", {})),
            "instances_on_device": json.dumps(payload.get("instances_on_device", {})),
            "last_heartbeat_ts": time.time()
        }
        await self.store.client.hset(worker_key, mapping=redis_hash)

    async def _prune_loop(self):
        while not self._shutdown_event.is_set():
            await self.prune_stale_workers()
            try:
                await asyncio.wait_for(self._shutdown_event.wait(), timeout=self.prune_interval)
            except asyncio.TimeoutError:
                pass

    async def prune_stale_workers(self) -> None:
        logger.debug("Pruning stale workers...")
        try:
            all_workers = await self.get_all_worker_info()
            current_time = time.time()
            stale_workers_count = 0

            for worker in all_workers:
                last_heartbeat = worker.get("last_heartbeat_ts", 0)
                if (current_time - float(last_heartbeat)) > self.worker_timeout:
                    node_id = worker["node_id"]
                    logger.warning(
                        f"Pruning stale worker {node_id}. Last heartbeat was "
                        f"{current_time - float(last_heartbeat):.2f} seconds ago."
                    )
                    await self.store.delete_worker(node_id)
                    stale_workers_count += 1
            
            if stale_workers_count > 0:
                logger.info(f"Pruned {stale_workers_count} stale worker(s).")

        except Exception as e:
            logger.error(f"Error during worker pruning: {e}", exc_info=True)

    def _generate_instance_id(self, model_name: str, backend: str) -> str:
        unique_part = uuid.uuid4().hex[:8]
        return f"{model_name}-{backend}-{unique_part}"

    async def get_worker_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        return await self.store.get_worker(node_id)

    async def get_all_worker_info(self) -> List[Dict[str, Any]]:
        return await self.store.get_all_workers()
