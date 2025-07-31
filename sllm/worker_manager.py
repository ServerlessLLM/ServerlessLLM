# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Set

import aiohttp

from sllm.kv_store import RedisStore
from sllm.logger import init_logger
from sllm.utils import generate_name, post_json_with_retry

logger = init_logger(__name__)

DEFAULT_WORKER_TIMEOUT = 60
DEFAULT_PRUNE_INTERVAL = 120
DEFAULT_SCALING_LOOP_INTERVAL = 5


class WorkerManager:
    def __init__(
        self, store: RedisStore, config: Optional[Mapping[str, Any]] = None
    ):
        self.store = store
        self.config = config or {}

        self.worker_timeout = self.config.get(
            "worker_timeout", DEFAULT_WORKER_TIMEOUT
        )
        self.prune_interval = self.config.get(
            "prune_interval", DEFAULT_PRUNE_INTERVAL
        )
        self.scaling_loop_interval = self.config.get(
            "scaling_loop_interval", DEFAULT_SCALING_LOOP_INTERVAL
        )

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
            await asyncio.gather(
                *self._background_tasks, return_exceptions=True
            )
        logger.info("WorkerManager shutdown complete.")

    async def _scaling_loop(self):
        while not self._shutdown_event.is_set():
            logger.debug("Scaling loop running...")
            try:
                decision_keys = [
                    key.decode("utf-8")
                    async for key in self.store.client.scan_iter(
                        "scaling_decision:*"
                    )
                ]

                for key in decision_keys:
                    instances_required_str = await self.store.client.get(key)
                    if not instances_required_str:
                        continue

                    instances_required = int(instances_required_str)
                    _, model_name, backend = key.split(":", 2)

                    # Get current limbo instances to adjust instances_needed
                    limbo_up_instances = (
                        await self.store.get_limbo_up_instances(
                            model_name, backend
                        )
                    )
                    limbo_down_instances = (
                        await self.store.get_limbo_down_instances(
                            model_name, backend
                        )
                    )
                    limbo_up_count = len(limbo_up_instances)
                    limbo_down_count = len(limbo_down_instances)

                    # Calculate actual instances needed accounting for limbo
                    if instances_required > 0:
                        # Scale up: subtract limbo_up (already requested but not confirmed)
                        instances_needed = instances_required - limbo_up_count
                    else:
                        # Scale down: add limbo_down (already requested to stop but not confirmed)
                        instances_needed = instances_required + limbo_down_count

                    if instances_needed > 0:
                        await self._execute_scale_up(
                            model_name, backend, instances_needed
                        )
                    elif instances_needed < 0:
                        await self._execute_scale_down(
                            model_name, backend, abs(instances_needed)
                        )

                    await self.store.client.delete(key)

                await asyncio.sleep(self.scaling_loop_interval)

            except asyncio.CancelledError:
                logger.info("Scaling loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in scaling loop: {e}", exc_info=True)
                await asyncio.sleep(self.scaling_loop_interval)

    async def _execute_scale_up(
        self, model_name: str, backend: str, count: int
    ):
        logger.info(
            f"Executing scale-up for '{model_name}:{backend}' by {count} instance(s)."
        )
        model_config = await self.store.get_model(model_name, backend)
        if not model_config:
            logger.error(
                f"Cannot scale up: Model config for '{model_name}:{backend}' not found."
            )
            return

        all_workers = await self.get_all_worker_info()
        model_identifier = f"{model_name}:{backend}"

        if not all_workers:
            logger.warning(
                f"No workers available to scale up for {model_identifier}."
            )
            return

        # Filter out workers that are currently busy or have pending start requests
        eligible_workers = []
        limbo_up_instances = await self.store.get_limbo_up_instances(
            model_name, backend
        )

        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            running_instances = instances_on_device.get(model_identifier, [])

            # Check if any limbo_up instances belong to this worker (by node_id in instance_id)
            node_id = worker["node_id"]
            worker_pattern = f"{model_name}-{backend}-{node_id}-"
            has_pending_starts = any(
                instance_id.startswith(worker_pattern)
                for instance_id in limbo_up_instances
            )

            # Only include workers with no running instances and no pending starts
            if len(running_instances) == 0 and not has_pending_starts:
                eligible_workers.append(worker)
            else:
                logger.debug(
                    f"Excluding worker {node_id} from scaling: "
                    f"running={len(running_instances)}, pending_starts={has_pending_starts}"
                )

        logger.info(
            f"Found {len(eligible_workers)} eligible workers for scaling {model_identifier}"
        )

        if not eligible_workers:
            logger.warning(
                f"No eligible workers available for {model_identifier} - all are busy"
            )
            return

        successful_starts = 0
        for i in range(count):
            if not eligible_workers:
                logger.warning(
                    f"No more eligible workers available for {model_identifier}"
                )
                break

            target_worker = random.choice(eligible_workers)

            logger.info(
                f"Attempting to start instance on worker {target_worker['node_id']}"
            )
            instance_id, success = await self._send_start_instance_request(
                target_worker, model_config
            )
            if success:
                successful_starts += 1

                logger.info(
                    f"Adding instance {instance_id} to limbo_up for {model_name}:{backend}"
                )
                added = await self.store.add_limbo_up_instance(
                    model_name, backend, instance_id
                )
                if not added:
                    logger.warning(
                        f"Instance {instance_id} was already in limbo_up"
                    )
                else:
                    logger.info(
                        f"Successfully added {instance_id} to limbo_up"
                    )

                # NOTE: is this necessary?
                await self.store.set_worker_status(
                    model_name, backend, target_worker["node_id"], "busy"
                )
            else:
                logger.error(
                    f"Failed to start instance on worker {target_worker['node_id']}. Removing from eligible list."
                )
                eligible_workers.remove(target_worker)

        logger.info(
            f"Successfully started {successful_starts}/{count} instances for {model_identifier}"
        )

    async def _execute_scale_down(
        self, model_name: str, backend: str, count: int
    ):
        logger.info(
            f"Executing scale-down for '{model_name}:{backend}' by {count} instance(s)."
        )

        all_workers = await self.get_all_worker_info()
        model_identifier = f"{model_name}:{backend}"

        running_instances = []
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            if model_identifier in instances_on_device:
                for instance_id in instances_on_device[model_identifier]:
                    running_instances.append(
                        {"worker": worker, "instance_id": instance_id}
                    )

        if not running_instances:
            logger.warning(
                f"No running instances of {model_identifier} found to scale down."
            )
            return

        instances_to_stop = random.sample(
            running_instances, min(count, len(running_instances))
        )

        if len(instances_to_stop) == len(running_instances):
            queue_length = await self.get_queue_length(model_name, backend)
            if queue_length > 0:
                logger.info(
                    f"Received request to stop all instances but queue is not empty, leaving one instance alive."
                )
                instances_to_stop = instances_to_stop[:-1]

        successful_stops = 0
        for item in instances_to_stop:
            worker = item["worker"]
            instance_id = item["instance_id"]

            logger.info(
                f"Attempting to stop instance {instance_id} on worker {worker['node_id']}"
            )
            success = await self._send_stop_instance_request(
                worker, instance_id
            )
            if success:
                successful_stops += 1
                # Add specific instance to limbo_down tracking (atomically updates counter too)
                logger.info(
                    f"Adding instance {instance_id} to limbo_down for {model_name}:{backend}"
                )
                added = await self.store.add_limbo_down_instance(
                    model_name, backend, instance_id
                )
                if not added:
                    logger.warning(
                        f"Instance {instance_id} was already in limbo_down"
                    )
                else:
                    logger.info(
                        f"Successfully added {instance_id} to limbo_down"
                    )
                # Worker status will be updated when heartbeat confirms instance is stopped
            else:
                logger.error(
                    f"Failed to stop instance {instance_id} on worker {worker['node_id']}"
                )

        logger.info(
            f"Successfully stopped {successful_stops}/{len(instances_to_stop)} instances for {model_identifier}"
        )

    async def count_running_instances(self, model_identifier: str) -> int:
        all_workers = await self.store.get_all_workers()
        count = 0
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            count += len(instances_on_device.get(model_identifier, []))
        return count

    async def _send_start_instance_request(
        self, worker: Dict[str, Any], model_config: Dict[str, Any]
    ) -> tuple[str, bool]:
        node_ip = worker.get("node_ip")
        if not node_ip:
            logger.error(
                f"Cannot send command to worker {worker['node_id']}: missing IP address."
            )
            return "", False

        # Generate instance_id at the head
        model_name = model_config.get("model")
        backend = model_config.get("backend")
        instance_id = self._generate_instance_id(
            model_name, backend, worker["node_id"]
        )

        url = f"http://{node_ip}:8001/start_instance"
        try:
            await post_json_with_retry(
                session=self.http_session,
                url=url,
                payload={
                    "model_config": model_config,
                    "instance_id": instance_id,
                },
                max_retries=3,
                timeout=300.0,
            )
            logger.info(
                f"Successfully sent start instance command to {worker['node_id']} with instance_id {instance_id}."
            )
            return instance_id, True
        except Exception as e:
            logger.error(
                f"HTTP request to start instance on {worker['node_id']} failed: {e}"
            )
            return instance_id, False

    async def _send_stop_instance_request(
        self, worker: Dict[str, Any], instance_id: str
    ) -> bool:
        node_ip = worker.get("node_ip")
        if not node_ip:
            logger.error(
                f"Cannot send command to worker {worker['node_id']}: missing IP address."
            )
            return False

        url = f"http://{node_ip}:8001/stop_instance"
        payload = {"instance_id": instance_id}
        try:
            await post_json_with_retry(
                session=self.http_session,
                url=url,
                payload=payload,
                max_retries=3,
                timeout=120.0,
            )
            logger.info(
                f"Successfully sent stop command for {instance_id} to {worker['node_id']}."
            )
            return True
        except Exception as e:
            logger.error(
                f"HTTP request to stop instance on {worker['node_id']} failed: {e}"
            )
            return False

    async def process_heartbeat(self, payload: Dict[str, Any]) -> None:
        node_id = payload.get("node_id")
        node_ip = payload.get("node_ip")
        node_port = payload.get("node_port", 8001)

        if not node_ip:
            logger.warning("Received a heartbeat with no node_ip. Ignoring.")
            return

        hardware_info = payload.get("hardware_info", {})
        instances_on_device = payload.get("instances_on_device", {})

        if not isinstance(instances_on_device, dict):
            logger.warning(f"Invalid instances_on_device format from {node_id}")
            instances_on_device = {}

        # Derive registered_models from instances_on_device keys
        registered_models = list(instances_on_device.keys())

        if not node_id:
            existing_worker = await self._find_worker_by_ip(node_ip)
            if existing_worker:
                await self._relink_worker(
                    node_ip,
                    node_port,
                    existing_worker,
                    hardware_info,
                    instances_on_device,
                )
            else:
                await self._register_worker(
                    node_ip,
                    node_port,
                    hardware_info,
                    instances_on_device,
                )
            return

        existing_worker = await self.store.get_worker(node_id)

        if existing_worker:
            existing_ip = existing_worker.get("node_ip")
            if existing_ip != node_ip:
                ip_conflict_worker = await self._find_worker_by_ip(node_ip)
                if (
                    ip_conflict_worker
                    and ip_conflict_worker["node_id"] != node_id
                ):
                    logger.info(
                        f"IP {node_ip} registered to different worker {ip_conflict_worker['node_id']}, resetting worker and proceeding with {node_id}"
                    )
                    await self._cleanup_worker_from_all_sets(
                        ip_conflict_worker["node_id"]
                    )
                    await self.store.delete_worker(
                        ip_conflict_worker["node_id"]
                    )

                logger.info(
                    f"Worker {node_id} IP changed from {existing_ip} to {node_ip}"
                )
                await self._send_confirmation_with_instances(
                    node_ip, node_port, node_id, existing_worker
                )
                return

            worker_status = await self._get_worker_status(node_id)
            if worker_status == "initializing":
                logger.debug(
                    f"Worker {node_id} is initializing, skipping heartbeat"
                )
                return
        else:
            existing_ip_worker = await self._find_worker_by_ip(node_ip)
            if existing_ip_worker:
                logger.warning(
                    f"Worker with unrecognized node_id '{node_id}' from registered IP {node_ip}. "
                    f"Expected node_id '{existing_ip_worker['node_id']}'. Resetting worker."
                )
                await self._relink_worker(
                    node_ip,
                    node_port,
                    existing_ip_worker,
                    hardware_info,
                    instances_on_device,
                )
                return
            else:
                await self._register_worker_with_id(
                    node_id,
                    node_ip,
                    node_port,
                    hardware_info,
                    instances_on_device,
                )
                return

        try:
            heartbeat_data = {
                "hardware_info": hardware_info,
                "instances_on_device": instances_on_device,
            }

            success = await self.store.atomic_worker_heartbeat_update(
                node_id, heartbeat_data
            )

            if not success:
                new_worker = {
                    "node_id": node_id,
                    "node_ip": node_ip,
                    "hardware_info": hardware_info,
                    "instances_on_device": instances_on_device,
                    "last_heartbeat_time": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }

                ip_to_node_key = f"ip_to_node:{node_ip}"
                (
                    registration_success,
                    existing_node,
                ) = await self.store.atomic_worker_registration(
                    new_worker, ip_to_node_key
                )

                if not registration_success:
                    logger.info(
                        f"IP {node_ip} registered to different worker {existing_node}, resetting worker and retrying registration for {node_id}"
                    )
                    await self._cleanup_worker_from_all_sets(existing_node)
                    await self.store.delete_worker(existing_node)

                    (
                        retry_success,
                        _,
                    ) = await self.store.atomic_worker_registration(
                        new_worker, ip_to_node_key
                    )

                    if not retry_success:
                        logger.error(
                            f"Failed to register worker {node_id} after cleaning up conflicting worker"
                        )
                        return

                logger.info(f"Successfully registered new worker {node_id}")

            worker_key = self.store._get_worker_key(node_id)
            async with self.store.client.pipeline(transaction=True) as pipe:
                pipe.hset(
                    worker_key,
                    mapping={
                        "hardware_info": json.dumps(hardware_info),
                        "instances_on_device": json.dumps(instances_on_device),
                        "last_heartbeat_time": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "status": "ready",
                    },
                )
                await pipe.execute()

            await self._update_worker_state_sets(node_id, instances_on_device)

            logger.debug(f"Processed heartbeat from worker {node_id}")

        except Exception as e:
            logger.error(
                f"Error processing heartbeat from {node_id}: {e}", exc_info=True
            )

    async def _update_worker_state_sets(
        self,
        node_id: str,
        instances_on_device: Dict[str, List[str]],
    ) -> None:
        # Get the previous state to detect changes
        try:
            previous_worker = await self.store.get_worker(node_id)
            previous_instances = (
                previous_worker.get("instances_on_device", {})
                if previous_worker
                else {}
            )
        except Exception:
            previous_instances = {}

        registered_models = list(instances_on_device.keys())
        for model_identifier in registered_models:
            try:
                model_name, backend = model_identifier.split(":", 1)
            except ValueError:
                logger.warning(
                    f"Invalid model identifier format: {model_identifier}"
                )
                continue

            current_instances = set(
                instances_on_device.get(model_identifier, [])
            )
            previous_instances_set = set(
                previous_instances.get(model_identifier, [])
            )

            # Log instance changes for debugging and update limbo tracking
            new_instances = current_instances - previous_instances_set
            if new_instances:
                logger.debug(
                    f"Detected {len(new_instances)} new instances on {node_id}: {new_instances}"
                )
                # Check if any of these new instances were in limbo_up and remove them
                limbo_up_instances = await self.store.get_limbo_up_instances(
                    model_name, backend
                )
                confirmed_starts = new_instances.intersection(
                    limbo_up_instances
                )
                for instance_id in confirmed_starts:
                    removed = await self.store.remove_limbo_up_instance(
                        model_name, backend, instance_id
                    )
                    if removed:
                        logger.info(
                            f"Instance {instance_id} started successfully - removed from limbo_up for {model_name}:{backend}"
                        )
                    else:
                        logger.warning(
                            f"Instance {instance_id} started but was not found in limbo_up for {model_name}:{backend}"
                        )

            stopped_instances = previous_instances_set - current_instances
            if stopped_instances:
                logger.debug(
                    f"Detected {len(stopped_instances)} stopped instances on {node_id}: {stopped_instances}"
                )
                # Check if any of these stopped instances were in limbo_down and remove them
                limbo_down_instances = (
                    await self.store.get_limbo_down_instances(
                        model_name, backend
                    )
                )
                confirmed_stops = stopped_instances.intersection(
                    limbo_down_instances
                )
                for instance_id in confirmed_stops:
                    logger.info(
                        f"Removing instance {instance_id} from limbo_down for {model_name}:{backend}"
                    )
                    removed = await self.store.remove_limbo_down_instance(
                        model_name, backend, instance_id
                    )
                    if removed:
                        logger.info(
                            f"Successfully removed {instance_id} from limbo_down"
                        )
                    else:
                        logger.warning(
                            f"Failed to remove {instance_id} from limbo_down"
                        )
                    if removed:
                        logger.debug(
                            f"Confirmed stop of instance {instance_id} for {model_identifier}"
                        )
                    else:
                        logger.warning(
                            f"Instance {instance_id} was not found in limbo_down for removal"
                        )

            running_instances = instances_on_device.get(model_identifier, [])

            if running_instances:
                await self.store.set_worker_status(
                    model_name, backend, node_id, "busy"
                )
            else:
                await self.store.set_worker_status(
                    model_name, backend, node_id, "ready"
                )

    async def _cleanup_worker_from_sets(
        self, node_id: str, model_name: str, backend: str
    ) -> None:
        ready_set = self.store._get_worker_set_key(model_name, backend, "ready")
        busy_set = self.store._get_worker_set_key(model_name, backend, "busy")

        await self.store.client.srem(ready_set, node_id)
        await self.store.client.srem(busy_set, node_id)

    async def _cleanup_worker_from_all_sets(self, node_id: str) -> None:
        ready_keys = [
            key.decode("utf-8")
            async for key in self.store.client.scan_iter("workers:ready:*")
        ]
        busy_keys = [
            key.decode("utf-8")
            async for key in self.store.client.scan_iter("workers:busy:*")
        ]

        all_keys = ready_keys + busy_keys
        if all_keys:
            pipe = self.store.client.pipeline()
            for key in all_keys:
                pipe.srem(key, node_id)
            await pipe.execute()

        # Clean up any limbo data for this worker
        await self.store.cleanup_worker_limbo_data(node_id)

    async def _prune_loop(self):
        while not self._shutdown_event.is_set():
            await self.prune_stale_workers()
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=self.prune_interval
                )
            except asyncio.TimeoutError:
                pass

    async def prune_stale_workers(self) -> None:
        logger.debug("Pruning stale workers...")
        try:
            all_workers = await self.get_all_worker_info()
            current_time = time.time()
            stale_workers_count = 0

            for worker in all_workers:
                last_heartbeat_str = worker.get("last_heartbeat_time")
                if last_heartbeat_str:
                    try:
                        if isinstance(last_heartbeat_str, str):
                            last_heartbeat = datetime.fromisoformat(
                                last_heartbeat_str.replace("Z", "+00:00")
                            ).timestamp()
                        else:
                            last_heartbeat = float(last_heartbeat_str)
                    except (ValueError, TypeError):
                        last_heartbeat = 0
                else:
                    last_heartbeat = 0

                if (current_time - last_heartbeat) > self.worker_timeout:
                    node_id = worker["node_id"]
                    logger.warning(
                        f"Pruning stale worker {node_id}. Last heartbeat was "
                        f"{current_time - last_heartbeat:.2f} seconds ago."
                    )

                    await self._cleanup_worker_from_all_sets(node_id)

                    await self.store.delete_worker(node_id)
                    stale_workers_count += 1

            if stale_workers_count > 0:
                logger.info(f"Pruned {stale_workers_count} stale worker(s).")

        except Exception as e:
            logger.error(f"Error during worker pruning: {e}", exc_info=True)

    async def graceful_worker_shutdown(self, node_id: str) -> bool:
        try:
            worker = await self.get_worker_info(node_id)
            if not worker:
                logger.warning(
                    f"Worker {node_id} not found for graceful shutdown"
                )
                return False

            instances_on_device = worker.get("instances_on_device", {})

            stop_tasks = []
            for model_identifier, instance_ids in instances_on_device.items():
                for instance_id in instance_ids:
                    task = self._send_stop_instance_request(worker, instance_id)
                    stop_tasks.append(task)

            if stop_tasks:
                logger.info(
                    f"Stopping {len(stop_tasks)} instances on worker {node_id}"
                )
                await asyncio.gather(*stop_tasks, return_exceptions=True)

            await self._cleanup_worker_from_all_sets(node_id)
            await self.store.delete_worker(node_id)

            logger.info(f"Gracefully shut down worker {node_id}")
            return True

        except Exception as e:
            logger.error(
                f"Error during graceful shutdown of worker {node_id}: {e}",
                exc_info=True,
            )
            return False

    async def get_worker_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        return await self.store.get_worker(node_id)

    async def get_all_worker_info(self) -> List[Dict[str, Any]]:
        return await self.store.get_all_workers()

    async def get_workers_for_model(
        self, model_name: str, backend: str, state: str = "ready"
    ) -> List[str]:
        worker_set_key = self.store._get_worker_set_key(
            model_name, backend, state
        )
        workers = await self.store.client.smembers(worker_set_key)
        return [worker.decode("utf-8") for worker in workers]

    async def _find_worker_by_ip(
        self, ip_address: str
    ) -> Optional[Dict[str, Any]]:
        all_workers = await self.store.get_all_workers()
        for worker in all_workers:
            if worker.get("node_ip") == ip_address:
                return worker
        return None

    async def _generate_unique_worker_id(self) -> str:
        max_attempts = 100
        for _ in range(max_attempts):
            potential_id = generate_name()
            existing_worker = await self.store.get_worker(potential_id)
            if not existing_worker:
                return potential_id

        return f"worker-{str(uuid.uuid4())[:8]}"

    def _generate_instance_id(
        self, model: str, backend: str, node_id: str = None
    ) -> str:
        unique_part = uuid.uuid4().hex[:8]
        if node_id:
            return f"{model}-{backend}-{node_id}-{unique_part}"
        else:
            return f"{model}-{backend}-{unique_part}"

    async def _set_worker_status(self, node_id: str, status: str) -> None:
        try:
            worker_key = self.store._get_worker_key(node_id)
            await self.store.client.hset(worker_key, "status", status)
            logger.debug(f"Set worker {node_id} status to {status}")
        except Exception as e:
            logger.error(
                f"Failed to set worker {node_id} status to {status}: {e}"
            )

    async def _get_worker_status(self, node_id: str) -> str:
        try:
            worker_key = self.store._get_worker_key(node_id)
            status = await self.store.client.hget(worker_key, "status")
            return status.decode("utf-8") if status else "ready"
        except Exception as e:
            logger.error(f"Failed to get worker {node_id} status: {e}")
            return "ready"

    async def _send_confirmation(
        self, worker_ip: str, worker_port: int, node_id: str
    ) -> None:
        try:
            confirmation_url = f"http://{worker_ip}:{worker_port}/confirmation"
            payload = {"node_id": node_id}

            await post_json_with_retry(
                session=self.http_session,
                url=confirmation_url,
                payload=payload,
                max_retries=2,
                timeout=5.0,
            )
            logger.info(
                f"Successfully sent confirmation to worker at {worker_ip}:{worker_port}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send confirmation to worker at {worker_ip}:{worker_port}: {e}"
            )

    async def _send_confirmation_with_instances(
        self,
        worker_ip: str,
        worker_port: int,
        node_id: str,
        existing_worker: Dict[str, Any],
    ) -> None:
        try:
            await self._set_worker_status(node_id, "initializing")
            await self._send_confirmation(worker_ip, worker_port, node_id)

            instances_on_device = self._parse_instances_from_worker(
                existing_worker
            )
            restarted_count = await self._restart_worker_instances(
                worker_ip, node_id, instances_on_device
            )

            logger.info(
                f"Successfully sent confirmation and restarted {restarted_count} instances on worker at {worker_ip}"
            )
            await self._set_worker_status(node_id, "ready")

        except Exception as e:
            logger.error(
                f"Failed to send confirmation with instances to worker at {worker_ip}: {e}"
            )
            await self._set_worker_status(node_id, "ready")

    async def get_queue_length(self, model_name: str, backend: str) -> int:
        return await self.store.get_queue_length(model_name, backend)

    @staticmethod
    def get_queue_name(model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    async def _register_worker(
        self,
        node_ip: str,
        node_port: int,
        hardware_info: Dict[str, Any],
        instances_on_device: Dict[str, List[str]],
    ) -> None:
        try:
            node_id = await self._generate_unique_worker_id()
            await self._send_confirmation(node_ip, node_port, node_id)
            logger.info(
                f"Generated new worker ID {node_id} for {node_ip}:{node_port}"
            )
        except Exception as e:
            logger.error(
                f"Error registering new worker at {node_ip}:{node_port}: {e}",
                exc_info=True,
            )

    async def _register_worker_with_id(
        self,
        node_id: str,
        node_ip: str,
        node_port: int,
        hardware_info: Dict[str, Any],
        instances_on_device: Dict[str, List[str]],
    ) -> None:
        try:
            worker_key = self.store._get_worker_key(node_id)
            ip_to_node_key = f"ip_to_node:{node_ip}"

            existing_node_for_ip = await self.store.client.get(ip_to_node_key)
            if (
                existing_node_for_ip
                and existing_node_for_ip.decode("utf-8") != node_id
            ):
                conflicting_node_id = existing_node_for_ip.decode("utf-8")
                logger.info(
                    f"IP {node_ip} registered to different worker {conflicting_node_id}, resetting worker and proceeding with {node_id}"
                )
                await self._cleanup_worker_from_all_sets(conflicting_node_id)
                await self.store.delete_worker(conflicting_node_id)

            async with self.store.client.pipeline(transaction=True) as pipe:
                pipe.hset(
                    worker_key,
                    mapping={
                        "node_id": node_id,
                        "node_ip": node_ip,
                        "hardware_info": json.dumps(hardware_info),
                        "instances_on_device": json.dumps(instances_on_device),
                        "last_heartbeat_time": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "status": "ready",
                    },
                )
                pipe.set(ip_to_node_key, node_id)
                pipe.sadd(self.store._get_workers_index_key(), node_id)
                await pipe.execute()

            await self._update_worker_state_sets(node_id, instances_on_device)

            logger.info(
                f"Successfully registered worker {node_id} with existing ID"
            )

        except Exception as e:
            logger.error(
                f"Error registering worker {node_id} at {node_ip}:{node_port}: {e}",
                exc_info=True,
            )

    async def _relink_worker(
        self,
        node_ip: str,
        node_port: int,
        existing_worker: Dict[str, Any],
        hardware_info: Dict[str, Any],
        instances_on_device: Dict[str, List[str]],
    ) -> None:
        node_id = existing_worker["node_id"]

        try:
            await self._set_worker_status(node_id, "initializing")
            logger.info(
                f"Relinking worker {node_id} at {node_ip}:{node_port} and reinstantiating models"
            )

            # Clean up any stale limbo data for this worker since it's restarting
            cleanup_count = await self.store.cleanup_worker_limbo_data(node_id)
            if cleanup_count > 0:
                logger.info(
                    f"Cleaned up {cleanup_count} stale limbo instances for relinked worker {node_id}"
                )

            await self._send_confirmation(node_ip, node_port, node_id)
            logger.info(f"Sent confirmation to relinked worker {node_id}")

            stored_instances = self._parse_instances_from_worker(
                existing_worker
            )
            reinstantiated_count = await self._restart_worker_instances(
                node_ip, node_id, stored_instances
            )

            logger.info(
                f"Successfully relinked worker {node_id} and reinstantiated {reinstantiated_count} instances"
            )

            await self._update_worker_record(
                node_id,
                node_ip,
                hardware_info,
                instances_on_device,
            )
            await self._update_worker_state_sets(node_id, instances_on_device)

        except Exception as e:
            logger.error(
                f"Error relinking worker {node_id} at {node_ip}:{node_port}: {e}",
                exc_info=True,
            )
            await self._set_worker_status(node_id, "ready")

    def _parse_instances_from_worker(
        self, worker: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        instances_on_device = {}
        if "instances_on_device" in worker:
            try:
                instances_on_device = json.loads(worker["instances_on_device"])
            except (json.JSONDecodeError, TypeError):
                instances_on_device = {}
        return instances_on_device

    async def _restart_worker_instances(
        self,
        worker_ip: str,
        node_id: str,
        instances_on_device: Dict[str, List[str]],
    ) -> int:
        start_instance_url = f"http://{worker_ip}:8001/start_instance"
        restarted_count = 0

        for model_identifier, instance_list in instances_on_device.items():
            if not isinstance(instance_list, list):
                continue

            try:
                model_name, backend = model_identifier.split(":", 1)
            except ValueError:
                logger.warning(
                    f"Invalid model identifier format: {model_identifier}"
                )
                continue

            model_config = await self.store.get_model(model_name, backend)
            if not model_config:
                logger.warning(
                    f"Cannot reinstantiate {model_identifier}: model config not found in store"
                )
                continue

            for instance_id in instance_list:
                restart_payload = {
                    "model_config": model_config,
                    "instance_id": instance_id,
                }

                try:
                    await post_json_with_retry(
                        session=self.http_session,
                        url=start_instance_url,
                        payload=restart_payload,
                        max_retries=2,
                        timeout=30.0,
                    )
                    restarted_count += 1
                    logger.info(
                        f"Reinstantiated instance {instance_id} for model {model_identifier} on worker {node_id}"
                    )
                except Exception as instance_error:
                    logger.error(
                        f"Error reinstantiating instance {instance_id} on worker {node_id}: {instance_error}"
                    )

        return restarted_count

    async def _update_worker_record(
        self,
        node_id: str,
        node_ip: str,
        hardware_info: Dict[str, Any],
        instances_on_device: Dict[str, List[str]],
    ) -> None:
        worker_key = self.store._get_worker_key(node_id)
        async with self.store.client.pipeline(transaction=True) as pipe:
            pipe.hset(
                worker_key,
                mapping={
                    "node_ip": node_ip,
                    "hardware_info": json.dumps(hardware_info),
                    "instances_on_device": json.dumps(instances_on_device),
                    "last_heartbeat_time": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "status": "ready",
                },
            )
            await pipe.execute()
