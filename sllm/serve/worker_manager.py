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

from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger
from sllm.serve.utils import HTTPRetryError, post_json_with_retry

logger = init_logger(__name__)

DEFAULT_WORKER_TIMEOUT = 60
DEFAULT_PRUNE_INTERVAL = 15
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
                    instances_needed_str = await self.store.client.get(key)
                    if not instances_needed_str:
                        continue

                    instances_needed = int(instances_needed_str)
                    _, model_name, backend = key.split(":", 2)

                    logger.info(
                        f"Found scaling task for '{model_name}:{backend}': need {instances_needed} instances."
                    )

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

        # Find workers that can run this model
        eligible_workers = []
        for worker in all_workers:
            registered_models = worker.get("registered_models", [])
            if model_identifier in registered_models:
                eligible_workers.append(worker)

        if not eligible_workers:
            logger.warning(
                f"No eligible workers available to scale up for {model_identifier}."
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
            success = await self._send_start_instance_request(
                target_worker, model_config
            )
            if success:
                successful_starts += 1
                # Add worker to ready set
                await self.store.set_worker_status(
                    target_worker["node_id"], model_name, backend, "ready"
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

        # Safety check: don't stop all instances if queue has work
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
                # Remove worker from ready/busy sets
                await self._cleanup_worker_from_sets(
                    worker["node_id"], model_name, backend
                )

        logger.info(
            f"Successfully stopped {successful_stops}/{len(instances_to_stop)} instances for {model_identifier}"
        )

    async def count_running_instances(self, model_identifier: str) -> int:
        """Counts the total number of active instances for a given model identifier."""
        all_workers = await self.store.get_all_workers()
        count = 0
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            count += len(instances_on_device.get(model_identifier, []))
        return count

    async def _send_start_instance_request(
        self, worker: Dict[str, Any], model_config: Dict[str, Any]
    ) -> bool:
        node_ip = worker.get("node_ip")
        if not node_ip:
            logger.error(
                f"Cannot send command to worker {worker['node_id']}: missing IP address."
            )
            return False

        url = f"http://{node_ip}:8001/start_instance"
        payload = {"model_config": model_config.get("backend_config", {})}
        try:
            await post_json_with_retry(
                session=self.http_session,
                url=url,
                payload=payload,
                max_retries=3,
                timeout=30.0,
            )
            logger.info(
                f"Successfully sent start instance command to {worker['node_id']}."
            )
            return True
        except HTTPRetryError as e:
            logger.error(
                f"HTTP request to start instance on {worker['node_id']} failed: {e}"
            )
            return False

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
                timeout=30.0,
            )
            logger.info(
                f"Successfully sent stop command for {instance_id} to {worker['node_id']}."
            )
            return True
        except HTTPRetryError as e:
            logger.error(
                f"HTTP request to stop instance on {worker['node_id']} failed: {e}"
            )
            return False

    async def process_heartbeat(self, payload: Dict[str, Any]) -> None:
        """Process worker heartbeat with validation and state management."""
        import aiohttp

        from .utils import generate_name

        # Extract node_id and node_ip first
        node_id = payload.get("node_id")
        node_ip = payload.get("node_ip")

        if not node_ip:
            logger.warning("Received a heartbeat with no node_ip. Ignoring.")
            return

        # Handle instance ID generation and confirmation
        if not node_id:
            # Check if IP already exists in worker registry
            existing_worker = await self._find_worker_by_ip(node_ip)

            if existing_worker:
                # Reject registration - IP conflict without proper node_id
                logger.warning(
                    f"Registration rejected: IP {node_ip} already registered to worker {existing_worker['node_id']}. Provide valid node_id to reconnect."
                )
                return
            else:
                # Generate new unique instance ID for new worker
                node_id = await self._generate_unique_worker_id()
                await self._send_confirmation(node_ip, node_id)
                logger.info(f"Generated new worker ID {node_id} for {node_ip}")
                return

        # Validate provided node_id and handle reconnections
        existing_worker = await self.store.get_worker(node_id)
        if not existing_worker:
            logger.warning(
                f"Registration rejected: Invalid node_id {node_id} provided by {node_ip}"
            )
            return

        # Check if worker IP changed (worker moved/restarted)
        existing_ip = existing_worker.get("node_ip")
        if existing_ip != node_ip:
            # Security check: ensure new IP isn't already used by another worker
            ip_conflict_worker = await self._find_worker_by_ip(node_ip)
            if ip_conflict_worker and ip_conflict_worker["node_id"] != node_id:
                logger.warning(
                    f"Registration rejected: Worker {node_id} trying to use IP {node_ip} already registered to worker {ip_conflict_worker['node_id']}"
                )
                return

            logger.info(
                f"Worker {node_id} IP changed from {existing_ip} to {node_ip}, updating record and restarting instances"
            )
            await self._send_confirmation_with_instances(
                node_ip, node_id, existing_worker
            )
            return

        # Check if worker is in initializing state (prevent race conditions)
        worker_status = await self._get_worker_status(node_id)
        if worker_status == "initializing":
            logger.debug(
                f"Worker {node_id} is initializing, skipping heartbeat processing"
            )
            return

        # Validate and parse optional fields
        hardware_info = payload.get("hardware_info", {})
        instances_on_device = payload.get("instances_on_device", {})
        registered_models = payload.get("registered_models", [])

        # Validate instances_on_device structure
        if not isinstance(instances_on_device, dict):
            logger.warning(f"Invalid instances_on_device format from {node_id}")
            instances_on_device = {}

        # Validate registered_models
        if not isinstance(registered_models, list):
            logger.warning(f"Invalid registered_models format from {node_id}")
            registered_models = []

        try:
            # Try atomic heartbeat update first
            heartbeat_data = {
                "hardware_info": hardware_info,
                "instances_on_device": instances_on_device,
                "registered_models": registered_models,
            }

            success = await self.store.atomic_worker_heartbeat_update(
                node_id, heartbeat_data
            )
            if not success:
                # Worker doesn't exist, need to register it first
                logger.info(
                    f"Worker {node_id} not found, creating new worker registration"
                )

                # Create new Worker object
                from sllm.serve.schema import HardwareInfo, Worker

                # Create HardwareInfo object from hardware_info dict
                hardware_info_obj = HardwareInfo.model_validate(hardware_info)

                new_worker = Worker(
                    node_id=node_id,
                    node_ip=node_ip,
                    hardware_info=hardware_info_obj,
                    instances_on_device=instances_on_device,
                    last_heartbeat_time=datetime.now(timezone.utc),
                )

                # Use atomic worker registration to prevent race conditions
                ip_to_node_key = f"ip_to_node:{node_ip}"
                (
                    registration_success,
                    existing_node,
                ) = await self.store.atomic_worker_registration(
                    new_worker, ip_to_node_key
                )

                if not registration_success:
                    logger.warning(
                        f"Worker registration failed for {node_id}: IP {node_ip} already registered to {existing_node}"
                    )
                    return

                logger.info(f"Successfully registered new worker {node_id}")

            # Update additional fields using pipeline for atomicity
            worker_key = self.store._get_worker_key(node_id)
            async with self.store.client.pipeline(transaction=True) as pipe:
                pipe.hset(
                    worker_key,
                    mapping={
                        "hardware_info": json.dumps(hardware_info),
                        "instances_on_device": json.dumps(instances_on_device),
                        "registered_models": json.dumps(registered_models),
                        "status": "ready",  # Reset to ready after successful heartbeat
                    },
                )
                await pipe.execute()

            # Update worker state sets based on current instances
            await self._update_worker_state_sets(
                node_id, instances_on_device, registered_models
            )

            logger.debug(f"Processed heartbeat from worker {node_id}")

        except Exception as e:
            logger.error(
                f"Error processing heartbeat from {node_id}: {e}", exc_info=True
            )

    async def _update_worker_state_sets(
        self,
        node_id: str,
        instances_on_device: Dict[str, List[str]],
        registered_models: List[str],
    ) -> None:
        """Update Redis sets that track worker availability for each model."""

        # For each model the worker can run
        for model_identifier in registered_models:
            try:
                model_name, backend = model_identifier.split(":", 1)
            except ValueError:
                logger.warning(
                    f"Invalid model identifier format: {model_identifier}"
                )
                continue

            # Check if worker has running instances for this model
            running_instances = instances_on_device.get(model_identifier, [])

            if running_instances:
                # Worker is busy (has running instances)
                await self.store.set_worker_status(
                    node_id, model_name, backend, "busy"
                )
            else:
                # Worker is ready (no running instances)
                await self.store.set_worker_status(
                    node_id, model_name, backend, "ready"
                )

    async def _cleanup_worker_from_sets(
        self, node_id: str, model_name: str, backend: str
    ) -> None:
        """Remove worker from both ready and busy sets for a specific model."""
        ready_set = self.store._get_worker_set_key(model_name, backend, "ready")
        busy_set = self.store._get_worker_set_key(model_name, backend, "busy")

        await self.store.client.srem(ready_set, node_id)
        await self.store.client.srem(busy_set, node_id)

    async def _cleanup_worker_from_all_sets(self, node_id: str) -> None:
        """Remove worker from all model worker sets when worker is deregistered."""
        # Get all worker set keys
        ready_keys = [
            key.decode("utf-8")
            async for key in self.store.client.scan_iter("workers:ready:*")
        ]
        busy_keys = [
            key.decode("utf-8")
            async for key in self.store.client.scan_iter("workers:busy:*")
        ]

        # Remove worker from all sets
        all_keys = ready_keys + busy_keys
        if all_keys:
            pipe = self.store.client.pipeline()
            for key in all_keys:
                pipe.srem(key, node_id)
            await pipe.execute()

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
        """Prune stale workers and clean up all associated state."""
        logger.debug("Pruning stale workers...")
        try:
            all_workers = await self.get_all_worker_info()
            current_time = time.time()
            stale_workers_count = 0

            for worker in all_workers:
                # Handle both new and legacy field names
                last_heartbeat_str = worker.get("last_heartbeat_time")
                if last_heartbeat_str:
                    try:
                        last_heartbeat = datetime.fromisoformat(
                            last_heartbeat_str
                        ).timestamp()
                    except (ValueError, TypeError):
                        last_heartbeat = 0
                else:
                    # Legacy field name fallback
                    last_heartbeat = worker.get("last_heartbeat_ts", 0)
                if (current_time - float(last_heartbeat)) > self.worker_timeout:
                    node_id = worker["node_id"]
                    logger.warning(
                        f"Pruning stale worker {node_id}. Last heartbeat was "
                        f"{current_time - float(last_heartbeat):.2f} seconds ago."
                    )

                    # Clean up worker from all sets before deleting
                    await self._cleanup_worker_from_all_sets(node_id)

                    # Delete worker record
                    await self.store.delete_worker(node_id)
                    stale_workers_count += 1

            if stale_workers_count > 0:
                logger.info(f"Pruned {stale_workers_count} stale worker(s).")

        except Exception as e:
            logger.error(f"Error during worker pruning: {e}", exc_info=True)

    async def graceful_worker_shutdown(self, node_id: str) -> bool:
        """Handle graceful worker shutdown - stop all instances and cleanup."""
        try:
            worker = await self.get_worker_info(node_id)
            if not worker:
                logger.warning(
                    f"Worker {node_id} not found for graceful shutdown"
                )
                return False

            instances_on_device = worker.get("instances_on_device", {})

            # Send stop commands for all running instances
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

            # Clean up worker state
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
        """Get list of worker node_ids that are ready/busy for a specific model."""
        worker_set_key = self.store._get_worker_set_key(
            model_name, backend, state
        )
        workers = await self.store.client.smembers(worker_set_key)
        return [worker.decode("utf-8") for worker in workers]

    async def _find_worker_by_ip(
        self, ip_address: str
    ) -> Optional[Dict[str, Any]]:
        """Find an existing worker by IP address."""
        all_workers = await self.store.get_all_workers()
        for worker in all_workers:
            if worker.get("node_ip") == ip_address:
                return worker
        return None

    async def _generate_unique_worker_id(self) -> str:
        """Generate a unique worker ID using utils.generate_name()."""
        from .utils import generate_name

        max_attempts = 100
        for _ in range(max_attempts):
            potential_id = generate_name()
            # Check if this ID already exists
            existing_worker = await self.store.get_worker(potential_id)
            if not existing_worker:
                return potential_id

        # Fallback if we can't find a unique name
        import uuid

        return f"worker-{str(uuid.uuid4())[:8]}"

    async def _set_worker_status(self, node_id: str, status: str) -> None:
        """Set worker status (ready, busy, initializing)."""
        try:
            worker_key = self.store._get_worker_key(node_id)
            await self.store.client.hset(worker_key, "status", status)
            logger.debug(f"Set worker {node_id} status to {status}")
        except Exception as e:
            logger.error(
                f"Failed to set worker {node_id} status to {status}: {e}"
            )

    async def _get_worker_status(self, node_id: str) -> str:
        """Get worker status."""
        try:
            worker_key = self.store._get_worker_key(node_id)
            status = await self.store.client.hget(worker_key, "status")
            return status.decode("utf-8") if status else "ready"
        except Exception as e:
            logger.error(f"Failed to get worker {node_id} status: {e}")
            return "ready"

    async def _send_confirmation(self, worker_ip: str, node_id: str) -> None:
        """Send confirmation with generated node_id to worker."""
        try:
            confirmation_url = f"http://{worker_ip}/confirmation"
            payload = {"node_id": node_id}

            async with aiohttp.ClientSession() as session:
                await post_json_with_retry(
                    session=session,
                    url=confirmation_url,
                    payload=payload,
                    max_retries=2,
                    timeout=5.0,
                )
                logger.info(
                    f"Successfully sent confirmation to worker at {worker_ip}"
                )
        except HTTPRetryError as e:
            logger.error(
                f"Failed to send confirmation to worker at {worker_ip}: {e}"
            )

    async def _send_confirmation_with_instances(
        self, worker_ip: str, node_id: str, existing_worker: Dict[str, Any]
    ) -> None:
        """Send confirmation with node_id and restart instances for registered models."""
        import json as json_lib

        import aiohttp

        try:
            # Set worker to initializing state
            await self._set_worker_status(node_id, "initializing")

            # First send confirmation with node_id
            confirmation_url = f"http://{worker_ip}/confirmation"
            payload = {"node_id": node_id}

            async with aiohttp.ClientSession() as session:
                await post_json_with_retry(
                    session=session,
                    url=confirmation_url,
                    payload=payload,
                    max_retries=2,
                    timeout=5.0,
                )

            # Parse instances_on_device from the existing worker record
            instances_on_device = {}
            if "instances_on_device" in existing_worker:
                try:
                    instances_on_device = json_lib.loads(
                        existing_worker["instances_on_device"]
                    )
                except (json_lib.JSONDecodeError, TypeError):
                    instances_on_device = {}

            # Restart each instance by calling /start_instance endpoint
            start_instance_url = f"http://{worker_ip}/start_instance"
            restarted_instances = 0

            for model_name, instance_list in instances_on_device.items():
                if isinstance(instance_list, list):
                    for instance_id in instance_list:
                        # Parse model_name to extract backend if needed
                        # For now, assume default backend or extract from registered_models
                        backend = "default"  # This should be derived properly

                        model_config = {
                            "model_name": model_name,
                            "backend": backend,
                        }

                        restart_payload = {
                            "model_config": model_config,
                            "instance_id": instance_id,
                        }

                        try:
                            await post_json_with_retry(
                                session=session,
                                url=start_instance_url,
                                payload=restart_payload,
                                max_retries=2,
                                timeout=30.0,
                            )
                            restarted_instances += 1
                            logger.info(
                                f"Restarted instance {instance_id} for model {model_name} on worker {node_id}"
                            )
                        except HTTPRetryError as instance_error:
                            logger.error(
                                f"Error restarting instance {instance_id} on worker {node_id}: {instance_error}"
                            )

            logger.info(
                f"Successfully sent confirmation and restarted {restarted_instances} instances on worker at {worker_ip}"
            )

            # Reset worker status to ready after successful initialization
            await self._set_worker_status(node_id, "ready")

        except Exception as e:
            logger.error(
                f"Failed to send confirmation with instances to worker at {worker_ip}: {e}"
            )
            # Reset worker status on failure
            await self._set_worker_status(node_id, "ready")

    async def get_queue_length(self, model_name: str, backend: str) -> int:
        """Get current queue length for a model."""
        return await self.store.get_queue_length(model_name, backend)

    @staticmethod
    def get_queue_name(model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"
