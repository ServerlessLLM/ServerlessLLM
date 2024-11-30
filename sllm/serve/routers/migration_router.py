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
from typing import Dict

import ray

from sllm.serve.logger import init_logger

from ..inference_instance import start_instance
from ..utils import InstanceHandle, InstanceStatus
from .roundrobin_router import RoundRobinRouter

logger = init_logger(__name__)


class MigrationRouter(RoundRobinRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
    ) -> None:
        super().__init__(
            model_name, resource_requirements, backend, backend_config, router_config
        )
        self.migration_record = {}
        self.migration_delta = self.router_config.get("migration_delta", 20)

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.request_count_lock:
            self.request_count += 1

        instance_allocation = self.loop.create_future()
        await self.request_queue.put(instance_allocation)
        logger.info(f"Enqueued request for model {self.model_name}")

        instance_id = await instance_allocation
        logger.info(f"{request_data}, type: {type(request_data)}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_instances:
                logger.error(f"Instance {instance_id} not found")
                return {"error": "Instance not found"}
            instance = self.ready_instances[instance_id]
        # NOTE: `.remote(request_data)` does not work, don't know why.
        # Looks like a known issue:
        # https://github.com/ray-project/ray/issues/26283#issuecomment-1780691475
        if action == "generate":
            result = await instance.backend_instance.generate.remote(
                request_data=request_data
            )
            if "preempted" in result:
                logger.info(f"Preempted request: {result}")
                target_instance_id = self.migration_record.get(instance_id)
                if not target_instance_id:
                    logger.error(f"No target instance found for {instance_id}")
                    return {"error": "No target instance found"}
                target_instance = self.ready_instances[target_instance_id]
                logger.info(
                    f"Resuming request on target instance: {target_instance_id}"
                )
                if "max_tokens" in request_data:
                    request_data["max_tokens"] -= result["completed_tokens"]
                result = await target_instance.backend_instance.resume_generate.remote(
                    request_data=request_data,
                    current_output=result["current_output"],
                )

        elif action == "encode":
            result = await instance.backend_instance.encode.remote(
                request_data=request_data
            )
        else:
            result = {"error": "Invalid action"}
        logger.info(f"Finished processing request")
        await instance.add_requests(-1)
        async with self.request_count_lock:
            self.request_count -= 1
        return result

    async def execute_migration_plan(self, migration_plan):
        logger.info(f"Executing migration plan: {migration_plan}")
        source_instance = migration_plan.source_instance
        source_instance_id = source_instance.instance_id
        target_node_id = migration_plan.target_node_id
        # start the target_instance on the target node
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{target_node_id}": 0.1,
            },
        }
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # TODO: Add max_queue_length to target_instance
        target_instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=10,
            num_gpu=self.resource_requirements["num_gpus"],
        )

        ret = await self.model_loading_scheduler.mark_resource.remote(
            self.model_name, instance_id, target_node_id
        )
        if not ret:
            logger.error(
                f"Failed to mark resource for instance {instance_id} for model {self.model_name}"
            )
            return None

        await start_instance.options(
            resources={
                "worker_node": 0.1,
                f"worker_id_{target_node_id}": 0.1,
            }
        ).remote(instance_id, self.backend, self.backend_config, startup_config)
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        target_instance.backend_instance = ray.get_actor(instance_id)
        async with target_instance.lock:
            target_instance.ready = True
            target_instance.node_id = target_node_id
        logger.info(
            f"Initialized instance {instance_id} for model {self.model_name}"
        )
        await target_instance.backend_instance.init_backend.remote()
        logger.info(
            f"Initialized backend for instance {instance_id} for model {self.model_name}"
        )
        # migrate the tokens from the source_instance (if still running) to the target_instance
        if source_instance_id not in self.ready_instances:
            logger.info(f"Instance {source_instance_id} not found")
            target_instance.ready = False
            await target_instance.backend_instance.shutdown.remote()
            ray.kill(target_instance.backend_instance)
            return None
        source_instance = self.ready_instances[source_instance_id]
        migration_iter = 0
        n_previous_tokens = 0
        while True:
            logger.info(f"Migration iteration {migration_iter}")
            current_tokens = ray.get(
                source_instance.backend_instance.get_current_tokens.remote()
            )
            n_delta_tokens = len(current_tokens) - n_previous_tokens
            logger.info(
                f"Number of tokens: {current_tokens}, delta: {n_delta_tokens}"
            )
            n_previous_tokens = len(current_tokens)
            if not current_tokens or n_delta_tokens <= self.migration_delta:
                logger.info(
                    "Migration completed:"
                    f"{None if not current_tokens else len(current_tokens)} tokens"
                )
                break
            ray.get(
                target_instance.backend_instance.resume_kv_cache.remote(
                    current_tokens
                )
            )
            migration_iter += 1
            logger.info(f"Migration iteration {migration_iter} completed")

        logger.info(f"Migrated instance {source_instance_id} to {instance_id}")
        async with self.instance_management_lock:
            if source_instance_id not in self.ready_instances:
                # source_instance has been removed
                logger.error(f"Instance {instance_id} not found")
                target_instance.ready = False
                await target_instance.backend_instance.shutdown.remote()
                ray.kill(target_instance.backend_instance)
                return None
            _ = self.ready_instances.pop(source_instance_id)
            async with source_instance.lock:
                source_instance.status = False
            self.ready_instances[instance_id] = target_instance
            self.migration_record[source_instance_id] = instance_id
        logger.info(f"Instance {source_instance_id} removed")
        await source_instance.backend_instance.shutdown.remote()
        logger.info(f"Shutdown instance {source_instance_id}")
        ray.kill(source_instance.backend_instance)
        logger.info(f"Killed instance {source_instance_id}")
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, source_instance_id, self.resource_requirements
        )
        logger.info(f"Deallocated instance {source_instance_id}")
        return instance_id

    async def get_instance_status(self, instance_id: str) -> InstanceStatus:
        logger.info(f"Getting status for instance: {instance_id}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_instances:
                logger.info(f"Instance {instance_id} not found")
                return None
            instance = self.ready_instances[instance_id]
            logger.info(f"Instance: {instance}")
            instance_status = await instance.get_status()
            instance_status.model_name = self.model_name
            instance_status.num_current_tokens = len(
                await instance.backend_instance.get_current_tokens.remote()
            )
            logger.info(f"Instance status: {instance_status}")
            return instance_status
