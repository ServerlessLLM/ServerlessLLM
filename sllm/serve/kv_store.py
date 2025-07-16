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

import os
import asyncio
import json
import redis.asyncio as redis
from typing import Any, Dict, List, Optional
from sllm.serve.logger import init_logger

"""
- workermanager
- modelmanager
- model task queues
- pubsub channel
"""
TIMEOUT = 60

logger = init_logger(__name__)

class RedisStore:
    def __init__(self, host: str = 'localhost', port: int = 6379):
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        final_host = host if host is not None else redis_host
        self.pool = redis.ConnectionPool(host=final_host, port=port)
        self.client = redis.Redis(connection_pool=self.pool)

    async def close(self):
        await self.pool.disconnect()

    ### MODEL METHODS ###
    def _get_model_key(self, model_name: str, backend: str) -> str:
        return f"model:{model_name}:{backend}"

    def _reconstruct_model(self, redis_hash: Dict[str, str]) -> Dict[str, Any]:
        redis_hash['auto_scaling_config'] = json.loads(redis_hash['auto_scaling_config'])
        redis_hash['backend_config'] = json.loads(redis_hash['backend_config'])
        return redis_hash

    async def register_model(self, model_data: Dict[str, Any]) -> None:
        key = self._get_model_key(model_data['model_name'], model_data['backend'])
        model_data_to_store = model_data.copy()
        model_data_to_store["auto_scaling_config"] = json.dumps(model_data.get("auto_scaling_config", {}))
        model_data_to_store["backend_config"] = json.dumps(model_data.get("backend_config", {}))
        
        await self.client.hset(key, mapping=model_data_to_store)

    async def register_model(self, model: Model) -> None:
        key = self._get_model_key(model.model_name, model.backend)
        model_dict = model.model_dump()
        
        model_dict['backend_config'] = model.backend_config.model_dump_json()
        model_dict['auto_scaling_config'] = model.auto_scaling_config.model_dump_json()
        model_dict['instances'] = json.dumps(model.instances)
        
        await self.client.hset(key, mapping=model_dict)

    async def get_model(self, model_name: str, backend: str) -> Optional[Model]:
        key = self._get_model_key(model_name, backend)
        redis_hash = await self.client.hgetall(key)
        if not redis_hash:
            return None
        
        decoded_hash = {k.decode(): v.decode() for k, v in redis_hash.items()}
        return Model.model_validate(decoded_hash)

    async def get_all_models(self) -> List[Model]:
        model_keys = [key async for key in self.client.scan_iter("model:*")]
        if not model_keys:
            return []
        
        tasks = [self.client.hgetall(key) for key in model_keys]
        all_hashes = await asyncio.gather(*tasks)
        
        models = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                models.append(Model.model_validate(decoded_hash))
        return models

    async def delete_model(self, model_name: str, backend: str) -> None:
        key = self._get_model_key(model_name, backend)
        await self.client.delete(key)
        await self.client.delete(self._get_task_queue_key(model_name, backend))
        await self.client.delete(self._get_worker_set_key(model_name, backend, "ready"))
        await self.client.delete(self._get_worker_set_key(model_name, backend, "busy"))


    ### WORKER METHODS ###
    def _get_worker_key(self, node_id: str) -> str:
        return f"worker:{node_id}"

    def _reconstruct_worker(self, redis_hash: Dict[str, str]) -> Dict[str, Any]:
        redis_hash['hardware_info'] = json.loads(redis_hash['hardware_info'])
        redis_hash['instances_on_device'] = json.loads(redis_hash['instances_on_device'])
        if 'last_heartbeat_ts' in redis_hash:
             redis_hash['last_heartbeat_ts'] = float(redis_hash['last_heartbeat_ts'])
        return redis_hash

    async def register_worker(self, worker: Worker) -> None:
        key = self._get_worker_key(worker.node_id)
        worker_dict = worker.model_dump()

        worker_dict['hardware_info'] = worker.hardware_info.model_dump_json()
        worker_dict['instances_alive'] = json.dumps(worker.instances_alive)
        worker_dict['last_heartbeat_time'] = worker.last_heartbeat_time.isoformat()

        await self.client.hset(key, mapping=worker_dict)

    async def get_worker(self, node_id: str) -> Optional[Worker]:
        key = self._get_worker_key(node_id)
        redis_hash = await self.client.hgetall(key)
        if not redis_hash:
            return None
            
        decoded_hash = {k.decode(): v.decode() for k, v in redis_hash.items()}
        return Worker.model_validate(decoded_hash)

    async def get_all_workers(self) -> List[Worker]:
        worker_keys = [key async for key in self.client.scan_iter("worker:*")]
        if not worker_keys:
            return []
            
        tasks = [self.client.hgetall(key) for key in worker_keys]
        all_hashes = await asyncio.gather(*tasks)
        
        workers = []
        for h in all_hashes:
            if h:
                decoded_hash = {k.decode(): v.decode() for k, v in h.items()}
                workers.append(Worker.model_validate(decoded_hash))
        return workers

    async def delete_worker(self, node_id: str) -> None:
        key = self._get_worker_key(node_id)
        await self.client.delete(key)


    ### DYNAMIC WORKER STATE & HEARTBEAT ###
    def _get_worker_set_key(self, model_name: str, backend: str, state: str) -> str:
        # state can be "ready" or "busy"
        return f"workers:{state}:{model_name}:{backend}"

    async def set_worker_status(self, node_id: str, model_name: str, backend: str, state: str) -> None:
        """Atomically moves a worker from one state set to another."""
        if state == "ready":
            source_set = self._get_worker_set_key(model_name, backend, "busy")
            dest_set = self._get_worker_set_key(model_name, backend, "ready")
        elif state == "busy":
            source_set = self._get_worker_set_key(model_name, backend, "ready")
            dest_set = self._get_worker_set_key(model_name, backend, "busy")
        else:
            raise ValueError("State must be 'ready' or 'busy'")
        
        await self.client.smove(source_set, dest_set, node_id)

    async def get_ready_worker(self, model_name: str, backend: str) -> Optional[str]:
        """Gets a random ready worker for a model. Non-blocking."""
        ready_set = self._get_worker_set_key(model_name, backend, "ready")
        return await self.client.spop(ready_set)

    async def worker_heartbeat(self, node_id: str) -> None:
        key = self._get_worker_key(node_id)
        await self.client.hset(key, "last_heartbeat_time", datetime.now(timezone.utc).isoformat())


    ### TASK QUEUE METHODS ###
    def _get_task_queue_key(self, model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    async def enqueue_task(self, model_name: str, backend: str, task_data: Dict[str, Any]) -> None:
        """Adds a task to the model's work queue."""
        key = self._get_task_queue_key(model_name, backend)
        await self.client.lpush(key, json.dumps(task_data))

    async def dequeue_task(self, model_name: str, backend: str, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """Blocks until a task is available, then returns it."""
        key = self._get_task_queue_key(model_name, backend)
        result = await self.client.brpop([key], timeout)
        if result:
            return json.loads(result[1])
        return None

    async def dequeue_from_any(self, queue_keys: List[str], timeout: int = 0) -> Optional[tuple[str, Dict[str, Any]]]:
        if not queue_keys:
            await asyncio.sleep(1)
            return None
        result = await self.client.brpop(queue_keys, timeout)
        if result:
            queue_name = result[0]
            task_data = json.loads(result[1])
            return queue_name, task_data
        return None
    
    async def get_queue_length(self, model_name: str, backend: str) -> int:
        """Gets the current length of the task queue, useful for autoscaling."""
        key = self._get_task_queue_key(model_name, backend)
        return await self.client.llen(key)


    ### PUBSUB RESULT CHANNEL METHODS ###
    def _get_result_channel_key(self, task_id: str) -> str:
        return f"result-channel:{task_id}"

    async def publish_result(self, task_id: str, result_data: Dict[str, Any]) -> None:
        """Publishes the final result to the task's unique channel."""
        channel = self._get_result_channel_key(task_id)
        await self.client.publish(channel, json.dumps(result_data))

    async def subscribe_to_result(self, task_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribes to a task's result channel and yields the message."""
        channel_name = self._get_result_channel_key(task_id)
        async with self.client.pubsub() as pubsub:
            await pubsub.subscribe(channel_name)
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=TIMEOUT) 
                if message:
                    yield json.loads(message['data'])
                    break 
                # TODO: Handle timeout case if needed
