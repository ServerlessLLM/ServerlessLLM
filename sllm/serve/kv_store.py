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
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

from sllm.serve.logger import init_logger
from sllm.serve.schema import Model, Worker

logger = init_logger(__name__)

# Lua scripts for atomic operations
ATOMIC_WORKER_REGISTRATION_SCRIPT = """
local worker_key = KEYS[1]
local workers_index_key = KEYS[2]
local ip_to_node_key = KEYS[3]
local node_ip = ARGV[1]
local node_id = ARGV[2]
local worker_data = ARGV[3]

-- Check if IP is already registered to a different node
local existing_node = redis.call('GET', ip_to_node_key)
if existing_node and existing_node ~= node_id then
    return {1, existing_node}  -- Return error code 1 with existing node ID
end

-- Atomically register the worker
redis.call('HSET', worker_key, 'data', worker_data)
redis.call('SADD', workers_index_key, node_id)
redis.call('SET', ip_to_node_key, node_id)

return {0}  -- Success
"""

ATOMIC_MODEL_DELETION_SCRIPT = """
local model_key = KEYS[1]
local lock_key = KEYS[2]
local models_index_key = KEYS[3]
local status_index_key = KEYS[4]
local expected_status = ARGV[1]
local lock_value = ARGV[2]

-- Check if we have the deletion lock
local current_lock = redis.call('GET', lock_key)
if not current_lock or current_lock ~= lock_value then
    return {1, "Lock not held"}  -- Error code 1
end

-- Check current model status
local model_data = redis.call('HGET', model_key, 'status')
if not model_data then
    return {2, "Model not found"}  -- Error code 2
end

if model_data ~= expected_status then
    return {3, "Status mismatch"}  -- Error code 3
end

-- Atomically delete the model
redis.call('DEL', model_key)
redis.call('SREM', models_index_key, model_key)
redis.call('SREM', status_index_key, model_key)
redis.call('DEL', lock_key)

return {0}  -- Success
"""

"""
- workermanager
- modelmanager
- model task queues
- pubsub channel
"""
TIMEOUT = 60


class RedisStore:
    def __init__(
        self, host: str = "localhost", port: int = 8008, max_retries: int = 3
    ):
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        final_host = host if host is not None else redis_host
        self.host = final_host
        self.port = port
        self.max_retries = max_retries
        self._connection_lock = threading.Lock()
        self._last_connection_attempt = 0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

        # Thread-safe deletion locks tracking
        self._deletion_locks = {}
        self._deletion_locks_lock = threading.Lock()

        # Initialize connection pool with health check settings
        self.pool = redis.ConnectionPool(
            host=final_host,
            port=port,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self.client = redis.Redis(connection_pool=self.pool)

    async def _ensure_connection(self) -> bool:
        """Ensure Redis connection is healthy, reconnect if needed."""
        try:
            await self.client.ping()
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                f"Redis connection lost: {e}, attempting to reconnect"
            )
            return await self._reconnect()

    async def _reconnect(self) -> bool:
        """Attempt to reconnect to Redis with exponential backoff."""
        with self._connection_lock:
            current_time = time.time()

            # Rate limit reconnection attempts
            if (
                current_time - self._last_connection_attempt
                < self._reconnect_delay
            ):
                return False

            self._last_connection_attempt = current_time

            try:
                # Close existing connection
                await self.pool.disconnect()

                # Create new connection pool
                self.pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                self.client = redis.Redis(connection_pool=self.pool)

                # Test the connection
                await self.client.ping()

                logger.info("Successfully reconnected to Redis")
                self._reconnect_delay = (
                    1.0  # Reset delay on successful connection
                )
                return True

            except Exception as e:
                logger.error(f"Failed to reconnect to Redis: {e}")
                # Exponential backoff for reconnection delay
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
                return False

    async def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute Redis operation with connection retry logic."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Ensure connection is healthy
                if not await self._ensure_connection():
                    raise ConnectionError("Redis connection unavailable")

                # Execute the operation
                return await operation(*args, **kwargs)

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                logger.warning(
                    f"Redis operation failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )

                if attempt < self.max_retries:
                    await asyncio.sleep(
                        0.5 * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    break
            except Exception as e:
                # Non-connection errors should not be retried
                raise e

        logger.error(
            f"Redis operation failed after {self.max_retries + 1} attempts: {last_exception}"
        )
        raise last_exception

    async def close(self):
        await self.pool.disconnect()

    ### MODEL METHODS ###
    def _get_model_key(self, model_name: str, backend: str) -> str:
        return f"model:{model_name}:{backend}"

    def _get_models_index_key(self) -> str:
        """Master index set containing all model keys."""
        return "models:all"

    def _get_model_status_index_key(self, status: str) -> str:
        """Index set for models by status."""
        return f"models:by_status:{status}"

    def _reconstruct_model(self, redis_hash: Dict[str, str]) -> Dict[str, Any]:
        redis_hash["auto_scaling_config"] = json.loads(
            redis_hash["auto_scaling_config"]
        )
        redis_hash["backend_config"] = json.loads(redis_hash["backend_config"])

        # Handle instances field properly
        if "instances" in redis_hash:
            try:
                redis_hash["instances"] = json.loads(redis_hash["instances"])
            except (json.JSONDecodeError, TypeError):
                redis_hash["instances"] = []

        return redis_hash

    async def register_model(self, model: Model) -> None:
        key = self._get_model_key(model.model_name, model.backend)
        model_dict = model.model_dump()

        model_dict["backend_config"] = model.backend_config.model_dump_json()
        model_dict["auto_scaling_config"] = (
            model.auto_scaling_config.model_dump_json()
        )
        model_dict["instances"] = json.dumps([])
        model_dict["status"] = "alive"

        # Use pipeline for atomic registration with index updates
        async with self.client.pipeline(transaction=True) as pipe:
            # Register the model
            pipe.hset(key, mapping=model_dict)
            # Add to master index
            pipe.sadd(self._get_models_index_key(), key)
            # Add to status index
            pipe.sadd(self._get_model_status_index_key("alive"), key)
            await pipe.execute()

    async def get_model(self, model_name: str, backend: str) -> Optional[Model]:
        key = self._get_model_key(model_name, backend)
        redis_hash = await self._execute_with_retry(self.client.hgetall, key)
        if not redis_hash:
            return None

        decoded_hash = {k.decode(): v.decode() for k, v in redis_hash.items()}
        return Model.model_validate(decoded_hash)

    async def get_all_models(self) -> List[Model]:
        """Get all models using efficient index-based lookup with pipeline."""
        # Get all model keys from the master index (O(1) operation)
        model_keys = await self._execute_with_retry(
            self.client.smembers, self._get_models_index_key()
        )

        if not model_keys:
            return []

        # Decode keys from bytes
        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in model_keys
        ]

        # Use pipeline for efficient bulk retrieval
        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        # Process results
        models = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_model(decoded_hash)
                    models.append(Model.model_validate(reconstructed))
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct model from hash {decoded_hash}: {e}"
                    )

        return models

    async def get_models_by_status(self, status: str) -> List[Model]:
        """Get models by status using efficient index lookup."""
        model_keys = await self._execute_with_retry(
            self.client.smembers, self._get_model_status_index_key(status)
        )

        if not model_keys:
            return []

        # Decode keys from bytes
        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in model_keys
        ]

        # Use pipeline for efficient bulk retrieval
        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        # Process results
        models = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_model(decoded_hash)
                    models.append(Model.model_validate(reconstructed))
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct model from hash {decoded_hash}: {e}"
                    )

        return models

    async def count_models_by_status(self, status: str) -> int:
        """Get count of models by status (O(1) operation)."""
        return await self._execute_with_retry(
            self.client.scard, self._get_model_status_index_key(status)
        )

    async def delete_model(self, model_key: str) -> None:
        # Use pipeline for atomic status update with index maintenance
        async with self.client.pipeline(transaction=True) as pipe:
            # Update status to excommunicado
            pipe.hset(model_key, "status", "excommunicado")
            # Move from alive to excommunicado index
            pipe.srem(self._get_model_status_index_key("alive"), model_key)
            pipe.sadd(
                self._get_model_status_index_key("excommunicado"), model_key
            )
            # Publish notification
            message = {"model_key": model_key}
            pipe.publish("model:delete:notifications", json.dumps(message))
            await pipe.execute()

    async def delete_lora_adapters(
        self,
        model_name: str,
        backend: str = "transformers",
        lora_adapters: List[str] = [],
    ):
        pass

    ### WORKER METHODS ###
    def _get_worker_key(self, node_id: str) -> str:
        return f"worker:{node_id}"

    def _get_workers_index_key(self) -> str:
        """Master index set containing all worker keys."""
        return "workers:all"

    def _get_workers_status_index_key(self, status: str) -> str:
        """Index set for workers by status."""
        return f"workers:by_status:{status}"

    def _reconstruct_worker(self, redis_hash: Dict[str, str]) -> Dict[str, Any]:
        redis_hash["hardware_info"] = json.loads(redis_hash["hardware_info"])
        redis_hash["instances_on_device"] = json.loads(
            redis_hash["instances_on_device"]
        )
        if "last_heartbeat_time" in redis_hash:
            # Convert ISO format back to datetime for schema compatibility
            redis_hash["last_heartbeat_time"] = datetime.fromisoformat(
                redis_hash["last_heartbeat_time"]
            )
        # Handle legacy field name for backward compatibility
        elif "last_heartbeat_ts" in redis_hash:
            redis_hash["last_heartbeat_time"] = datetime.fromtimestamp(
                float(redis_hash["last_heartbeat_ts"]), tz=timezone.utc
            )
        return redis_hash

    async def register_worker(self, worker: Worker) -> None:
        key = self._get_worker_key(worker.node_id)
        worker_dict = worker.model_dump()

        worker_dict["hardware_info"] = worker.hardware_info.model_dump_json()
        worker_dict["instances_on_device"] = json.dumps(
            worker.instances_on_device
        )
        worker_dict["last_heartbeat_time"] = (
            worker.last_heartbeat_time.isoformat()
        )

        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=worker_dict)
            pipe.sadd(self._get_workers_index_key(), key)
            pipe.sadd(self._get_workers_status_index_key("ready"), key)
            await pipe.execute()

    async def get_worker(self, node_id: str) -> Optional[Worker]:
        key = self._get_worker_key(node_id)
        redis_hash = await self.client.hgetall(key)
        if not redis_hash:
            return None

        decoded_hash = {k.decode(): v.decode() for k, v in redis_hash.items()}
        return Worker.model_validate(decoded_hash)

    async def get_all_workers(self) -> List[Worker]:
        """Get all workers using efficient index-based lookup with pipeline."""
        worker_keys = await self._execute_with_retry(
            self.client.smembers, self._get_workers_index_key()
        )

        if not worker_keys:
            return []

        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in worker_keys
        ]

        # Use pipeline for efficient bulk retrieval
        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        # Process results
        workers = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_worker(decoded_hash)
                    workers.append(Worker.model_validate(reconstructed))
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct worker from hash {decoded_hash}: {e}"
                    )

        return workers

    async def get_workers_by_status(self, status: str) -> List[Worker]:
        """Get workers by status using efficient index lookup."""
        worker_keys = await self._execute_with_retry(
            self.client.smembers, self._get_workers_status_index_key(status)
        )

        if not worker_keys:
            return []

        # Decode keys from bytes
        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in worker_keys
        ]

        # Use pipeline for efficient bulk retrieval
        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        # Process results
        workers = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_worker(decoded_hash)
                    workers.append(Worker.model_validate(reconstructed))
                except Exception as e:
                    logger.warning(
                        f"Failed to reconstruct worker from hash {decoded_hash}: {e}"
                    )

        return workers

    async def count_workers_by_status(self, status: str) -> int:
        """Get count of workers by status (O(1) operation)."""
        return await self._execute_with_retry(
            self.client.scard, self._get_workers_status_index_key(status)
        )

    async def get_active_workers(self) -> List[Worker]:
        """Get all active workers (ready + busy + initializing)."""
        # Use union of status sets for active workers
        ready_workers = await self.get_workers_by_status("ready")
        busy_workers = await self.get_workers_by_status("busy")
        initializing_workers = await self.get_workers_by_status("initializing")

        # Combine and deduplicate by node_id
        all_workers = ready_workers + busy_workers + initializing_workers
        seen_ids = set()
        unique_workers = []
        for worker in all_workers:
            if worker.node_id not in seen_ids:
                unique_workers.append(worker)
                seen_ids.add(worker.node_id)

        return unique_workers

    async def delete_worker(self, node_id: str) -> None:
        key = self._get_worker_key(node_id)
        # Use pipeline for atomic deletion with index cleanup
        async with self.client.pipeline(transaction=True) as pipe:
            # Delete the worker data
            pipe.delete(key)
            # Remove from master index
            pipe.srem(self._get_workers_index_key(), key)
            # Remove from all status indexes
            pipe.srem(self._get_workers_status_index_key("ready"), key)
            pipe.srem(self._get_workers_status_index_key("busy"), key)
            pipe.srem(self._get_workers_status_index_key("initializing"), key)
            await pipe.execute()

    ### DYNAMIC WORKER STATE & HEARTBEAT ###
    def _get_worker_set_key(
        self, model_name: str, backend: str, state: str
    ) -> str:
        # state can be "ready" or "busy"
        return f"workers:{state}:{model_name}:{backend}"

    async def set_worker_status(
        self, node_id: str, model_name: str, backend: str, state: str
    ) -> None:
        """Atomically moves a worker from one state set to another."""
        if state == "ready":
            source_set = self._get_worker_set_key(model_name, backend, "busy")
            dest_set = self._get_worker_set_key(model_name, backend, "ready")
        elif state == "busy":
            source_set = self._get_worker_set_key(model_name, backend, "ready")
            dest_set = self._get_worker_set_key(model_name, backend, "busy")
        else:
            raise ValueError("State must be 'ready' or 'busy'")

        await self._execute_with_retry(
            self.client.smove, source_set, dest_set, node_id
        )

    async def get_ready_worker(
        self, model_name: str, backend: str
    ) -> Optional[str]:
        """Gets a random ready worker for a model. Non-blocking."""
        ready_set = self._get_worker_set_key(model_name, backend, "ready")
        return await self.client.spop(ready_set)

    async def worker_heartbeat(self, node_id: str) -> None:
        key = self._get_worker_key(node_id)
        await self.client.hset(
            key, "last_heartbeat_time", datetime.now(timezone.utc).isoformat()
        )

    ### TASK QUEUE METHODS ###
    def _get_task_queue_key(self, model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    async def enqueue_task(
        self, model_name: str, backend: str, task_data: Dict[str, Any]
    ) -> None:
        """Adds a task to the model's work queue."""
        key = self._get_task_queue_key(model_name, backend)
        await self._execute_with_retry(
            self.client.lpush, key, json.dumps(task_data)
        )

    async def dequeue_task(
        self, model_name: str, backend: str, timeout: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Blocks until a task is available, then returns it."""
        key = self._get_task_queue_key(model_name, backend)
        result = await self._execute_with_retry(
            self.client.brpop, [key], timeout
        )
        if result:
            return json.loads(result[1])
        return None

    async def dequeue_from_any(
        self, queue_keys: List[str], timeout: int = 0
    ) -> Optional[tuple[str, Dict[str, Any]]]:
        if not queue_keys:
            await asyncio.sleep(1)
            return None
        result = await self._execute_with_retry(
            self.client.brpop, queue_keys, timeout
        )
        if result:
            queue_name = result[0]
            task_data = json.loads(result[1])
            return queue_name, task_data
        return None

    async def get_queue_length(self, model_name: str, backend: str) -> int:
        key = self._get_task_queue_key(model_name, backend)
        return await self.client.llen(key)

    async def acquire_deletion_lock(
        self, model_name: str, backend: str, timeout: int = 300
    ) -> bool:
        """Acquire exclusive lock for model deletion to prevent race conditions."""
        lock_key = f"deletion_lock:{model_name}:{backend}"
        lock_value = str(uuid.uuid4())
        acquired = await self.client.set(
            lock_key, lock_value, nx=True, ex=timeout
        )
        if acquired:
            with self._deletion_locks_lock:
                self._deletion_locks[f"{model_name}:{backend}"] = lock_value
            return True
        return False

    async def release_deletion_lock(
        self, model_name: str, backend: str
    ) -> bool:
        """Release deletion lock, but only if we own it."""
        lock_key = f"deletion_lock:{model_name}:{backend}"

        with self._deletion_locks_lock:
            lock_value = self._deletion_locks.get(f"{model_name}:{backend}")

        if not lock_value:
            return False

        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        result = await self.client.eval(lua_script, 1, lock_key, lock_value)
        if result:
            with self._deletion_locks_lock:
                self._deletion_locks.pop(f"{model_name}:{backend}", None)
        return bool(result)

    ### PUBSUB RESULT CHANNEL METHODS ###
    def _get_result_channel_key(self, task_id: str) -> str:
        return f"result-channel:{task_id}"

    async def publish_result(
        self, task_id: str, result_data: Dict[str, Any]
    ) -> None:
        """Publishes the final result to the task's unique channel with TTL."""
        channel = self._get_result_channel_key(task_id)

        # Use pipeline to atomically publish and set TTL
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.publish(channel, json.dumps(result_data))
            # Set a marker key with TTL to track channel lifetime
            pipe.setex(f"{channel}:ttl", 600, "1")  # 10 minutes TTL
            await pipe.execute()

    async def subscribe_to_result(
        self, task_id: str, timeout: int = TIMEOUT
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribes to a task's result channel and yields the message."""
        channel_name = self._get_result_channel_key(task_id)
        async with self.client.pubsub() as pubsub:
            await pubsub.subscribe(channel_name)
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=timeout
                )
                if message:
                    yield json.loads(message["data"])
                    break
                else:
                    # Handle timeout - check if channel TTL expired
                    ttl_key = f"{channel_name}:ttl"
                    ttl_exists = await self.client.exists(ttl_key)
                    if not ttl_exists:
                        # Channel expired, return timeout error
                        yield {
                            "error": {
                                "code": "TaskTimeout",
                                "message": f"Task {task_id} result channel expired",
                            }
                        }
                        break

    ### PERFORMANCE AND MONITORING METHODS ###
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics using efficient O(1) operations."""
        # Use pipeline for atomic metric collection
        async with self.client.pipeline() as pipe:
            # Model metrics
            pipe.scard(self._get_models_index_key())
            pipe.scard(self._get_model_status_index_key("alive"))
            pipe.scard(self._get_model_status_index_key("excommunicado"))

            # Worker metrics
            pipe.scard(self._get_workers_index_key())
            pipe.scard(self._get_workers_status_index_key("ready"))
            pipe.scard(self._get_workers_status_index_key("busy"))
            pipe.scard(self._get_workers_status_index_key("initializing"))

            results = await pipe.execute()

        return {
            "models": {
                "total": results[0],
                "alive": results[1],
                "excommunicado": results[2],
            },
            "workers": {
                "total": results[3],
                "ready": results[4],
                "busy": results[5],
                "initializing": results[6],
                "active": results[4] + results[5] + results[6],
            },
        }

    async def get_total_queue_length(self) -> int:
        """Get total length of all task queues."""
        # Get all queue keys for registered models
        all_models = await self.get_models_by_status("alive")
        total_length = 0

        if all_models:
            async with self.client.pipeline() as pipe:
                for model in all_models:
                    queue_key = self._get_task_queue_key(
                        model.model_name, model.backend
                    )
                    pipe.llen(queue_key)

                queue_lengths = await pipe.execute()
                total_length = sum(queue_lengths)

        return total_length

    async def cleanup_expired_result_channels(self) -> int:
        """Clean up expired result channel TTL markers."""
        # Find all TTL marker keys that have expired
        ttl_keys = []
        async for key in self.client.scan_iter("result-channel:*:ttl"):
            ttl_keys.append(key.decode() if isinstance(key, bytes) else key)

        expired_count = 0
        if ttl_keys:
            # Check which ones have expired (TTL = -1 means expired/missing)
            async with self.client.pipeline() as pipe:
                for key in ttl_keys:
                    pipe.ttl(key)
                ttl_values = await pipe.execute()

            # Remove expired TTL markers
            expired_keys = [
                key for key, ttl in zip(ttl_keys, ttl_values) if ttl == -1
            ]
            if expired_keys:
                await self.client.delete(*expired_keys)
                expired_count = len(expired_keys)

        return expired_count

    ### ATOMIC OPERATIONS WITH LUA SCRIPTS ###
    async def atomic_model_status_update(
        self, model_key: str, old_status: str, new_status: str
    ) -> bool:
        """Atomically update model status with index maintenance using Lua script."""
        lua_script = """
        local model_key = KEYS[1]
        local old_status = ARGV[1]
        local new_status = ARGV[2]
        local old_index = KEYS[2]
        local new_index = KEYS[3]

        -- Check if model exists and has expected status
        local current_status = redis.call("HGET", model_key, "status")
        if current_status ~= old_status then
            return 0  -- Status mismatch, operation failed
        end

        -- Update status and maintain indexes atomically
        redis.call("HSET", model_key, "status", new_status)
        redis.call("SREM", old_index, model_key)
        redis.call("SADD", new_index, model_key)

        return 1  -- Success
        """

        old_index = self._get_model_status_index_key(old_status)
        new_index = self._get_model_status_index_key(new_status)

        result = await self.client.eval(
            lua_script,
            3,
            model_key,
            old_index,
            new_index,
            old_status,
            new_status,
        )
        return bool(result)

    async def atomic_worker_transition(
        self, node_id: str, from_state: str, to_state: str
    ) -> bool:
        """Atomically transition worker between states using Lua script."""
        lua_script = """
        local worker_key = KEYS[1]
        local from_index = KEYS[2]
        local to_index = KEYS[3]
        local node_id = ARGV[1]
        local to_state = ARGV[2]

        -- Check if worker exists and is in expected state
        local is_in_from_state = redis.call("SISMEMBER", from_index, worker_key)
        if is_in_from_state == 0 then
            return 0  -- Worker not in expected state
        end

        -- Atomic state transition
        redis.call("SMOVE", from_index, to_index, worker_key)
        redis.call("HSET", worker_key, "status", to_state)

        return 1  -- Success
        """

        worker_key = self._get_worker_key(node_id)
        from_index = self._get_workers_status_index_key(from_state)
        to_index = self._get_workers_status_index_key(to_state)

        result = await self.client.eval(
            lua_script, 3, worker_key, from_index, to_index, node_id, to_state
        )
        return bool(result)

    async def atomic_worker_heartbeat_update(
        self, node_id: str, heartbeat_data: Dict[str, Any]
    ) -> bool:
        """Atomically update worker heartbeat with timestamp using Lua script."""
        lua_script = """
        local worker_key = KEYS[1]
        local workers_index = KEYS[2]
        local timestamp = ARGV[1]

        -- Check if worker exists
        local exists = redis.call("EXISTS", worker_key)
        if exists == 0 then
            return 0  -- Worker doesn't exist
        end

        -- Update heartbeat timestamp atomically
        redis.call("HSET", worker_key, "last_heartbeat_time", timestamp)

        -- Ensure worker is in master index
        redis.call("SADD", workers_index, worker_key)

        return 1  -- Success
        """

        worker_key = self._get_worker_key(node_id)
        workers_index = self._get_workers_index_key()
        timestamp = datetime.now(timezone.utc).isoformat()

        result = await self.client.eval(
            lua_script, 2, worker_key, workers_index, timestamp
        )
        return bool(result)

    async def atomic_worker_registration(
        self, worker: Worker, ip_to_node_key: str
    ) -> tuple[bool, Optional[str]]:
        """Atomically register a worker using Lua script to prevent race conditions."""
        worker_key = self._get_worker_key(worker.node_id)
        workers_index_key = self._get_workers_index_key()

        # Serialize worker data
        worker_dict = worker.model_dump()
        worker_dict["hardware_info"] = worker.hardware_info.model_dump_json()
        worker_dict["instances_on_device"] = json.dumps(
            worker.instances_on_device
        )
        worker_dict["last_heartbeat_time"] = (
            worker.last_heartbeat_time.isoformat()
        )
        worker_data = json.dumps(worker_dict)

        result = await self.client.eval(
            ATOMIC_WORKER_REGISTRATION_SCRIPT,
            3,
            worker_key,
            workers_index_key,
            ip_to_node_key,
            worker.node_ip,
            worker.node_id,
            worker_data,
        )

        if result[0] == 0:
            # Also add to status index
            await self.client.sadd(
                self._get_workers_status_index_key("ready"), worker_key
            )
            return True, None
        else:
            return False, result[1]  # Return existing node ID

    async def atomic_model_deletion(
        self, model_key: str, expected_status: str, lock_value: str
    ) -> tuple[bool, Optional[str]]:
        """Atomically delete a model using Lua script to prevent race conditions."""
        lock_key = f"deletion_lock:{model_key}"
        models_index_key = self._get_models_index_key()
        status_index_key = self._get_model_status_index_key(expected_status)

        result = await self.client.eval(
            ATOMIC_MODEL_DELETION_SCRIPT,
            4,
            model_key,
            lock_key,
            models_index_key,
            status_index_key,
            expected_status,
            lock_value,
        )

        if result[0] == 0:
            return True, None
        else:
            return False, result[1]  # Return error message
