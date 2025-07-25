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

logger = init_logger(__name__)

ATOMIC_WORKER_REGISTRATION_SCRIPT = """
local worker_key = KEYS[1]
local workers_index_key = KEYS[2]
local ip_to_node_key = KEYS[3]
local status_index_key = KEYS[4]
local node_ip = ARGV[1]
local node_id = ARGV[2]
local worker_data = ARGV[3]

local existing_node = redis.call('GET', ip_to_node_key)
if existing_node and existing_node ~= node_id then
    return {1, existing_node}
end

local worker_fields = cjson.decode(worker_data)
for field, value in pairs(worker_fields) do
    redis.call('HSET', worker_key, field, value)
end
redis.call('SADD', workers_index_key, worker_key)
redis.call('SET', ip_to_node_key, node_id)
redis.call('SADD', status_index_key, worker_key)

return {0}
"""

ATOMIC_MODEL_DELETION_SCRIPT = """
local model_key = KEYS[1]
local lock_key = KEYS[2]
local models_index_key = KEYS[3]
local status_index_key = KEYS[4]
local expected_status = ARGV[1]
local lock_value = ARGV[2]

local current_lock = redis.call('GET', lock_key)
if not current_lock or current_lock ~= lock_value then
    return {1, "Lock not held"}
end

local model_data = redis.call('HGET', model_key, 'status')
if not model_data then
    return {2, "Model not found"}
end

if model_data ~= expected_status then
    return {3, "Status mismatch"}
end

redis.call('DEL', model_key)
redis.call('SREM', models_index_key, model_key)
redis.call('SREM', status_index_key, model_key)
redis.call('DEL', lock_key)

return {0}
"""

TIMEOUT = 60


class RedisStore:
    def __init__(
        self, host: str = "localhost", port: int = 6379, max_retries: int = 3
    ):
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        final_host = host if host is not None else redis_host
        self.host = final_host
        self.port = port
        self.max_retries = max_retries
        self._connection_lock = threading.Lock()
        with self._connection_lock:
            self._last_connection_attempt = 0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 30.0

        self._deletion_locks = {}
        self._deletion_locks_lock = threading.Lock()
        self._lock_timestamps = {}

        self.pool = redis.ConnectionPool(
            host=final_host,
            port=port,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        self.client = redis.Redis(connection_pool=self.pool)

    async def _ensure_connection(self) -> bool:
        try:
            await self.client.ping()
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Redis connection lost: {e}")
            return await self._reconnect()

    async def _reconnect(self) -> bool:
        with self._connection_lock:
            current_time = time.time()

            if (
                current_time - self._last_connection_attempt
                < self._reconnect_delay
            ):
                return False

            self._last_connection_attempt = current_time

            try:
                await self.pool.disconnect()

                self.pool = redis.ConnectionPool(
                    host=self.host,
                    port=self.port,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                self.client = redis.Redis(connection_pool=self.pool)

                await self.client.ping()

                self._reconnect_delay = 1.0
                return True

            except Exception as e:
                logger.error(f"Failed to reconnect to Redis: {e}")
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )
                return False

    async def _execute_with_retry(self, operation, *args, **kwargs):
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if not await self._ensure_connection():
                    raise ConnectionError("Redis connection unavailable")

                return await operation(*args, **kwargs)

            except (ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt == self.max_retries:
                    logger.warning(
                        f"Redis operation failed after {self.max_retries + 1} attempts"
                    )

                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
                else:
                    break
            except Exception:
                raise

        logger.error(
            f"Redis operation failed after {self.max_retries + 1} attempts: {last_exception}"
        )
        raise last_exception

    async def close(self):
        await self.pool.disconnect()

    async def reset_store(self) -> bool:
        try:
            await self._execute_with_retry(self.client.flushdb)
            logger.info("Redis store reset successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reset Redis store: {e}")
            return False

    async def initialize_store(
        self, reset_on_start: bool = True, full_reset: bool = False
    ) -> bool:
        try:
            if not await self._ensure_connection():
                logger.error("Failed to connect to Redis during initialization")
                return False

            if reset_on_start:
                if full_reset:
                    success = await self.reset_store()
                    if not success:
                        return False
                else:
                    cleanup_stats = await self.cleanup_store_data()
                    logger.info(f"Selective cleanup completed: {cleanup_stats}")

            logger.info("Redis store initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Redis store: {e}")
            return False

    async def cleanup_store_data(
        self,
        cleanup_models: bool = True,
        cleanup_workers: bool = True,
        cleanup_queues: bool = True,
        cleanup_locks: bool = True,
    ) -> Dict[str, int]:
        cleanup_stats = {
            "models": 0,
            "workers": 0,
            "queues": 0,
            "locks": 0,
            "channels": 0,
        }

        try:
            if cleanup_models:
                model_keys = await self._execute_with_retry(
                    self.client.smembers, self._get_models_index_key()
                )
                if model_keys:
                    decoded_keys = [
                        key.decode() if isinstance(key, bytes) else key
                        for key in model_keys
                    ]
                    await self._execute_with_retry(
                        self.client.delete, *decoded_keys
                    )
                    await self._execute_with_retry(
                        self.client.delete, self._get_models_index_key()
                    )
                    await self._execute_with_retry(
                        self.client.delete,
                        self._get_model_status_index_key("alive"),
                        self._get_model_status_index_key("excommunicado"),
                    )
                    cleanup_stats["models"] = len(decoded_keys)

            if cleanup_workers:
                worker_keys = await self._execute_with_retry(
                    self.client.smembers, self._get_workers_index_key()
                )
                if worker_keys:
                    decoded_keys = [
                        key.decode() if isinstance(key, bytes) else key
                        for key in worker_keys
                    ]
                    await self._execute_with_retry(
                        self.client.delete, *decoded_keys
                    )
                    await self._execute_with_retry(
                        self.client.delete, self._get_workers_index_key()
                    )
                    await self._execute_with_retry(
                        self.client.delete,
                        self._get_workers_status_index_key("ready"),
                        self._get_workers_status_index_key("busy"),
                        self._get_workers_status_index_key("initializing"),
                    )
                    cleanup_stats["workers"] = len(decoded_keys)

            if cleanup_queues:
                queue_keys = []
                async for key in self.client.scan_iter("queue:*"):
                    queue_keys.append(
                        key.decode() if isinstance(key, bytes) else key
                    )
                async for key in self.client.scan_iter("workers:*"):
                    queue_keys.append(
                        key.decode() if isinstance(key, bytes) else key
                    )
                if queue_keys:
                    await self._execute_with_retry(
                        self.client.delete, *queue_keys
                    )
                    cleanup_stats["queues"] = len(queue_keys)

            if cleanup_locks:
                lock_keys = []
                async for key in self.client.scan_iter("deletion_lock:*"):
                    lock_keys.append(
                        key.decode() if isinstance(key, bytes) else key
                    )
                if lock_keys:
                    await self._execute_with_retry(
                        self.client.delete, *lock_keys
                    )
                    cleanup_stats["locks"] = len(lock_keys)

            channel_count = await self.cleanup_expired_result_channels()
            cleanup_stats["channels"] = channel_count

            with self._deletion_locks_lock:
                self._deletion_locks.clear()
                self._lock_timestamps.clear()

            logger.info(f"Store cleanup completed: {cleanup_stats}")
            return cleanup_stats

        except Exception as e:
            logger.error(f"Failed to cleanup store data: {e}")
            return cleanup_stats

    ### MODEL METHODS ###
    def _get_model_key(self, model_name: str, backend: str) -> str:
        return f"model:{model_name}:{backend}"

    def _get_models_index_key(self) -> str:
        return "models:all"

    def _get_model_status_index_key(self, status: str) -> str:
        return f"models:by_status:{status}"

    def _reconstruct_model(self, redis_hash: Dict[str, str]) -> Dict[str, Any]:
        redis_hash["auto_scaling_config"] = json.loads(
            redis_hash["auto_scaling_config"]
        )
        redis_hash["backend_config"] = json.loads(redis_hash["backend_config"])

        if "instances" in redis_hash:
            try:
                redis_hash["instances"] = json.loads(redis_hash["instances"])
            except (json.JSONDecodeError, TypeError):
                redis_hash["instances"] = []

        return redis_hash

    async def register_model(self, model: dict) -> None:
        model_name = model.get("model")
        if not model_name:
            raise ValueError("Model configuration must include 'model' key")
        key = self._get_model_key(model_name, model["backend"])
        model_dict = model.copy()
        model_dict["model"] = model_name

        if "backend_config" in model:
            backend_config = json.dumps(model["backend_config"])
            model_dict["backend_config"] = backend_config
            enable_lora = model["backend_config"].get("enable_lora", False)
            lora_adapters = model["backend_config"].get("lora_adapters", {})
        if "auto_scaling_config" in model:
            model_dict["auto_scaling_config"] = json.dumps(
                model["auto_scaling_config"]
            )
        model_dict["instances"] = json.dumps([])
        model_dict["status"] = "alive"

        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=model_dict)
            pipe.sadd(self._get_models_index_key(), key)
            pipe.sadd(self._get_model_status_index_key("alive"), key)
            # TODO: implement lora registration logic
            await pipe.execute()

    async def get_model(self, model_name: str, backend: str) -> Optional[dict]:
        key = self._get_model_key(model_name, backend)
        redis_hash = await self._execute_with_retry(self.client.hgetall, key)
        if not redis_hash:
            return None

        decoded_hash = {k.decode(): v.decode() for k, v in redis_hash.items()}
        return self._reconstruct_model(decoded_hash)

    async def get_all_models(self) -> List[dict]:
        model_keys = await self._execute_with_retry(
            self.client.smembers, self._get_models_index_key()
        )

        if not model_keys:
            return []

        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in model_keys
        ]

        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        models = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_model(decoded_hash)
                    models.append(reconstructed)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct model: {e}")

        return models

    async def get_models_by_status(self, status: str) -> List[dict]:
        model_keys = await self._execute_with_retry(
            self.client.smembers, self._get_model_status_index_key(status)
        )

        if not model_keys:
            return []

        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in model_keys
        ]

        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        models = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_model(decoded_hash)
                    models.append(reconstructed)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct model: {e}")

        return models

    async def count_models_by_status(self, status: str) -> int:
        return await self._execute_with_retry(
            self.client.scard, self._get_model_status_index_key(status)
        )

    async def delete_model(self, model_name: str, backend: str) -> None:
        model_key = self._get_model_key(model_name, backend)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(model_key, "status", "excommunicado")
            pipe.srem(self._get_model_status_index_key("alive"), model_key)
            pipe.sadd(
                self._get_model_status_index_key("excommunicado"), model_key
            )
            message = {"model_key": model_key}
            pipe.publish("model:delete:notifications", json.dumps(message))
            # TODO: delete lora adapters associated with the model too
            await pipe.execute()

    # TODO: make consistent with lora adapter storage in sllm
    async def delete_lora_adapters(
        self,
        model_name: str,
        backend: str = "transformers",
        lora_adapters: Optional[List[str]] = None,
    ):
        if lora_adapters is None:
            lora_adapters = []
        pass

    ### WORKER METHODS ###
    def _get_worker_key(self, node_id: str) -> str:
        return f"worker:{node_id}"

    def _get_workers_index_key(self) -> str:
        return "workers:all"

    def _get_workers_status_index_key(self, status: str) -> str:
        return f"workers:by_status:{status}"

    def _reconstruct_worker(self, redis_hash: Dict[str, str]) -> Dict[str, Any]:
        redis_hash["hardware_info"] = json.loads(redis_hash["hardware_info"])
        redis_hash["instances_on_device"] = json.loads(
            redis_hash["instances_on_device"]
        )
        return redis_hash

    async def register_worker(self, worker: dict) -> None:
        key = self._get_worker_key(worker["node_id"])
        worker_dict = worker.copy()

        worker_dict["hardware_info"] = json.dumps(worker["hardware_info"])
        worker_dict["instances_on_device"] = json.dumps(
            worker["instances_on_device"]
        )
        if isinstance(worker["last_heartbeat_time"], str):
            worker_dict["last_heartbeat_time"] = worker["last_heartbeat_time"]
        else:
            worker_dict["last_heartbeat_time"] = worker[
                "last_heartbeat_time"
            ].isoformat()

        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(key, mapping=worker_dict)
            pipe.sadd(self._get_workers_index_key(), key)
            pipe.sadd(self._get_workers_status_index_key("ready"), key)
            await pipe.execute()

    async def get_worker(self, node_id: str) -> Optional[dict]:
        key = self._get_worker_key(node_id)
        redis_hash = await self._execute_with_retry(self.client.hgetall, key)
        if not redis_hash:
            return None

        decoded_hash = {k.decode(): v.decode() for k, v in redis_hash.items()}
        return self._reconstruct_worker(decoded_hash)

    async def get_all_workers(self) -> List[dict]:
        worker_keys = await self._execute_with_retry(
            self.client.smembers, self._get_workers_index_key()
        )

        if not worker_keys:
            return []

        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in worker_keys
        ]

        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        workers = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_worker(decoded_hash)
                    workers.append(reconstructed)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct worker: {e}")

        return workers

    async def get_workers_by_status(self, status: str) -> List[dict]:
        worker_keys = await self._execute_with_retry(
            self.client.smembers, self._get_workers_status_index_key(status)
        )

        if not worker_keys:
            return []

        decoded_keys = [
            key.decode() if isinstance(key, bytes) else key
            for key in worker_keys
        ]

        async with self.client.pipeline() as pipe:
            for key in decoded_keys:
                pipe.hgetall(key)
            all_hashes = await pipe.execute()

        workers = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                try:
                    reconstructed = self._reconstruct_worker(decoded_hash)
                    workers.append(reconstructed)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct worker: {e}")

        return workers

    async def count_workers_by_status(self, status: str) -> int:
        return await self._execute_with_retry(
            self.client.scard, self._get_workers_status_index_key(status)
        )

    async def get_active_workers(self) -> List[dict]:
        ready_workers = await self.get_workers_by_status("ready")
        busy_workers = await self.get_workers_by_status("busy")
        initializing_workers = await self.get_workers_by_status("initializing")

        all_workers = ready_workers + busy_workers + initializing_workers
        unique_workers = {worker["node_id"]: worker for worker in all_workers}
        return list(unique_workers.values())

    async def delete_worker(self, node_id: str) -> None:
        key = self._get_worker_key(node_id)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.delete(key)
            pipe.srem(self._get_workers_index_key(), key)
            pipe.srem(self._get_workers_status_index_key("ready"), key)
            pipe.srem(self._get_workers_status_index_key("busy"), key)
            pipe.srem(self._get_workers_status_index_key("initializing"), key)
            await pipe.execute()

    ### DYNAMIC WORKER STATE & HEARTBEAT ###
    def _get_worker_set_key(
        self, model_name: str, backend: str, status: str
    ) -> str:
        return f"workers:{status}:{model_name}:{backend}"

    async def set_worker_status(
        self, model_name: str, backend: str, node_id: str, status: str
    ) -> None:
        if status == "ready":
            source_set = self._get_worker_set_key(model_name, backend, "busy")
            dest_set = self._get_worker_set_key(model_name, backend, "ready")
        elif status == "busy":
            source_set = self._get_worker_set_key(model_name, backend, "ready")
            dest_set = self._get_worker_set_key(model_name, backend, "busy")
        else:
            raise ValueError("Status must be 'ready' or 'busy'")

        await self._execute_with_retry(
            self.client.smove, source_set, dest_set, node_id
        )

    async def get_ready_worker(
        self, model_name: str, backend: str
    ) -> Optional[str]:
        ready_set = self._get_worker_set_key(model_name, backend, "ready")
        return await self._execute_with_retry(self.client.spop, ready_set)

    async def worker_heartbeat(
        self, node_id: str, heartbeat_data: Optional[dict] = None
    ) -> None:
        key = self._get_worker_key(node_id)
        if heartbeat_data:
            heartbeat_data_copy = heartbeat_data.copy()
            heartbeat_data_copy["last_heartbeat_time"] = datetime.now(
                timezone.utc
            ).isoformat()
            for field, value in heartbeat_data_copy.items():
                await self._execute_with_retry(
                    self.client.hset, key, field, value
                )
        else:
            await self._execute_with_retry(
                self.client.hset,
                key,
                "last_heartbeat_time",
                datetime.now(timezone.utc).isoformat(),
            )

    ### TASK QUEUE METHODS ###
    def _get_task_queue_key(self, model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    async def enqueue_task(
        self, model_name: str, backend: str, task_data: Dict[str, Any]
    ) -> None:
        key = self._get_task_queue_key(model_name, backend)
        await self._execute_with_retry(
            self.client.lpush, key, json.dumps(task_data)
        )

    async def dequeue_task(
        self, model_name: str, backend: str, timeout: int = 0
    ) -> Optional[Dict[str, Any]]:
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
            queue_name = (
                result[0].decode()
                if isinstance(result[0], bytes)
                else result[0]
            )
            task_data = json.loads(result[1])
            return queue_name, task_data
        return None

    async def get_queue_length(self, model_name: str, backend: str) -> int:
        key = self._get_task_queue_key(model_name, backend)
        return await self._execute_with_retry(self.client.llen, key)

    async def acquire_deletion_lock(
        self, model_name: str, backend: str, timeout: int = 300
    ) -> bool:
        lock_key = f"deletion_lock:{model_name}:{backend}"
        lock_value = str(uuid.uuid4())
        acquired = await self._execute_with_retry(
            self.client.set, lock_key, lock_value, nx=True, ex=timeout
        )
        if acquired:
            with self._deletion_locks_lock:
                lock_key_name = f"{model_name}:{backend}"
                self._deletion_locks[lock_key_name] = lock_value
                self._lock_timestamps[lock_key_name] = time.time()
            return True
        return False

    async def release_deletion_lock(
        self, model_name: str, backend: str
    ) -> bool:
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
        result = await self._execute_with_retry(
            self.client.eval, lua_script, 1, lock_key, lock_value
        )
        if result:
            with self._deletion_locks_lock:
                lock_key_name = f"{model_name}:{backend}"
                self._deletion_locks.pop(lock_key_name, None)
                self._lock_timestamps.pop(lock_key_name, None)
        return bool(result)

    ### PUBSUB RESULT CHANNEL METHODS ###
    def _get_result_channel_key(self, task_id: str) -> str:
        return f"result-channel:{task_id}"

    async def publish_result(
        self, task_id: str, result_data: Dict[str, Any]
    ) -> None:
        channel = self._get_result_channel_key(task_id)

        async with self.client.pipeline(transaction=True) as pipe:
            pipe.publish(channel, json.dumps(result_data))
            pipe.setex(f"{channel}:ttl", 600, "1")
            await pipe.execute()

    async def subscribe_to_result(
        self, task_id: str, timeout: int = TIMEOUT
    ) -> AsyncGenerator[Dict[str, Any], None]:
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
                    ttl_key = f"{channel_name}:ttl"
                    ttl_exists = await self._execute_with_retry(
                        self.client.exists, ttl_key
                    )
                    if not ttl_exists:
                        yield {
                            "error": {
                                "code": "TaskTimeout",
                                "message": f"Task {task_id} result channel expired",
                            }
                        }
                        break

    ### PERFORMANCE AND MONITORING METHODS ###
    async def get_performance_metrics(self) -> Dict[str, Any]:
        async with self.client.pipeline() as pipe:
            pipe.scard(self._get_models_index_key())
            pipe.scard(self._get_model_status_index_key("alive"))
            pipe.scard(self._get_model_status_index_key("excommunicado"))

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
        all_models = await self.get_models_by_status("alive")
        total_length = 0

        if all_models:
            async with self.client.pipeline() as pipe:
                for model in all_models:
                    queue_key = self._get_task_queue_key(
                        model.get("model"), model["backend"]
                    )
                    pipe.llen(queue_key)

                queue_lengths = await pipe.execute()
                total_length = sum(queue_lengths)

        return total_length

    async def cleanup_expired_result_channels(self) -> int:
        ttl_keys = []
        async for key in self.client.scan_iter("result-channel:*:ttl"):
            ttl_keys.append(key.decode() if isinstance(key, bytes) else key)

        expired_count = 0
        if ttl_keys:
            async with self.client.pipeline() as pipe:
                for key in ttl_keys:
                    pipe.ttl(key)
                ttl_values = await pipe.execute()

            expired_keys = [
                key for key, ttl in zip(ttl_keys, ttl_values) if ttl == -1
            ]
            if expired_keys:
                await self._execute_with_retry(
                    self.client.delete, *expired_keys
                )
                expired_count = len(expired_keys)

        return expired_count

    async def cleanup_all_expired(self) -> Dict[str, int]:
        channel_count = await self.cleanup_expired_result_channels()
        lock_count = await self.cleanup_expired_deletion_locks()
        return {"expired_channels": channel_count, "expired_locks": lock_count}

    async def cleanup_expired_deletion_locks(
        self, timeout_seconds: int = 3600
    ) -> int:
        current_time = time.time()
        expired_locks = []

        with self._deletion_locks_lock:
            for lock_key, timestamp in list(self._lock_timestamps.items()):
                if current_time - timestamp > timeout_seconds:
                    expired_locks.append(lock_key)

            for lock_key in expired_locks:
                self._deletion_locks.pop(lock_key, None)
                self._lock_timestamps.pop(lock_key, None)

        if expired_locks:
            redis_keys = [
                f"deletion_lock:{lock_key}" for lock_key in expired_locks
            ]
            try:
                await self._execute_with_retry(self.client.delete, *redis_keys)
            except Exception as e:
                logger.warning(f"Failed to cleanup expired deletion locks: {e}")

        return len(expired_locks)

    ### ATOMIC OPERATIONS WITH LUA SCRIPTS ###
    async def atomic_model_status_update(
        self, model_key: str, old_status: str, new_status: str
    ) -> bool:
        lua_script = """
        local model_key = KEYS[1]
        local old_status = ARGV[1]
        local new_status = ARGV[2]
        local old_index = KEYS[2]
        local new_index = KEYS[3]

        local current_status = redis.call("HGET", model_key, "status")
        if current_status ~= old_status then
            return 0
        end

        redis.call("HSET", model_key, "status", new_status)
        redis.call("SREM", old_index, model_key)
        redis.call("SADD", new_index, model_key)

        return 1
        """

        old_index = self._get_model_status_index_key(old_status)
        new_index = self._get_model_status_index_key(new_status)

        result = await self._execute_with_retry(
            self.client.eval,
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
        self, node_id: str, from_status: str, to_status: str
    ) -> bool:
        lua_script = """
        local worker_key = KEYS[1]
        local from_index = KEYS[2]
        local to_index = KEYS[3]
        local node_id = ARGV[1]
        local to_status = ARGV[2]

        local is_in_from_state = redis.call("SISMEMBER", from_index, worker_key)
        if is_in_from_state == 0 then
            return 0
        end

        redis.call("SMOVE", from_index, to_index, worker_key)
        redis.call("HSET", worker_key, "status", to_status)

        return 1
        """

        worker_key = self._get_worker_key(node_id)
        from_index = self._get_workers_status_index_key(from_status)
        to_index = self._get_workers_status_index_key(to_status)

        result = await self._execute_with_retry(
            self.client.eval,
            lua_script,
            3,
            worker_key,
            from_index,
            to_index,
            node_id,
            to_status,
        )
        return bool(result)

    async def atomic_worker_heartbeat_update(
        self, node_id: str, heartbeat_data: dict
    ) -> bool:
        lua_script = """
        local worker_key = KEYS[1]
        local workers_index = KEYS[2]
        local heartbeat_json = ARGV[1]

        local exists = redis.call("EXISTS", worker_key)
        if exists == 0 then
            return 0
        end

        local heartbeat_fields = cjson.decode(heartbeat_json)
        for field, value in pairs(heartbeat_fields) do
            redis.call("HSET", worker_key, field, value)
        end

        redis.call("SADD", workers_index, worker_key)

        return 1
        """

        worker_key = self._get_worker_key(node_id)
        workers_index = self._get_workers_index_key()

        heartbeat_data_copy = {}
        for key, value in heartbeat_data.items():
            if isinstance(value, (dict, list)):
                heartbeat_data_copy[key] = json.dumps(value)
            else:
                heartbeat_data_copy[key] = value
        heartbeat_data_copy["last_heartbeat_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        heartbeat_json = json.dumps(heartbeat_data_copy)

        result = await self._execute_with_retry(
            self.client.eval,
            lua_script,
            2,
            worker_key,
            workers_index,
            heartbeat_json,
        )
        return bool(result)

    async def atomic_worker_registration(
        self, worker: dict, ip_to_node_key: str
    ) -> tuple[bool, Optional[str]]:
        worker_key = self._get_worker_key(worker["node_id"])
        workers_index_key = self._get_workers_index_key()
        status_index_key = self._get_workers_status_index_key("ready")

        worker_dict = worker.copy()
        worker_dict["hardware_info"] = json.dumps(worker["hardware_info"])
        worker_dict["instances_on_device"] = json.dumps(
            worker["instances_on_device"]
        )
        if isinstance(worker["last_heartbeat_time"], str):
            worker_dict["last_heartbeat_time"] = worker["last_heartbeat_time"]
        else:
            worker_dict["last_heartbeat_time"] = worker[
                "last_heartbeat_time"
            ].isoformat()
        worker_data = json.dumps(worker_dict)

        result = await self._execute_with_retry(
            self.client.eval,
            ATOMIC_WORKER_REGISTRATION_SCRIPT,
            4,
            worker_key,
            workers_index_key,
            ip_to_node_key,
            status_index_key,
            worker["node_ip"],
            worker["node_id"],
            worker_data,
        )

        if result[0] == 0:
            return True, None
        else:
            return False, result[1]

    async def atomic_model_deletion(
        self, model_key: str, expected_status: str, lock_value: str
    ) -> tuple[bool, Optional[str]]:
        lock_key = f"deletion_lock:{model_key}"
        models_index_key = self._get_models_index_key()
        status_index_key = self._get_model_status_index_key(expected_status)

        result = await self._execute_with_retry(
            self.client.eval,
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
            return False, result[1]
