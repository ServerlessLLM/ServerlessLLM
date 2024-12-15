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
import logging
import uuid
from typing import Dict, Optional

import ray

from sllm.serve.inference_instance import start_instance

from .router_utils import InstanceHandle, SllmRouter

logger = logging.getLogger(__name__)


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 2)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


@ray.remote(num_cpus=1, resources={"control_node": 0.1})
class RoundRobinRouter(SllmRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
    ) -> None:
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        self.starting_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.auto_scaler = None
        logger.info(f"Created new handler for model {self.model_name}")

    async def start(self, auto_scaling_config: Dict[str, int]):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        async with self.auto_scaling_lock:
            self.auto_scaling_config = auto_scaling_config
        self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
        self.load_balancer = asyncio.create_task(self._load_balancer_loop())
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(self, auto_scaling_config: Dict[str, int]):
        async with self.auto_scaling_lock:
            self.auto_scaling_config = auto_scaling_config
        logger.info(
            f"Model {self.model_name}'s auto scaling config updated to {auto_scaling_config}"
        )

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.request_count_lock:
            self.request_count += 1

        async with self.idle_time_lock:
            self.idle_time = 0

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

    async def shutdown(self):
        async with self.running_lock:
            self.running = False
        # stop all instances
        # return all unfinished requests
        while not self.request_queue.empty():
            request_data, done_event = await self.request_queue.get()
            done_event.set_result({"error": "Instance cancelled"})

        async with self.instance_management_lock:
            deleted_instance_id = list(self.ready_instances.keys())
        delete_tasks = [
            self._shutdown_instance(instance_id)
            for instance_id in deleted_instance_id
        ]
        await asyncio.gather(*delete_tasks)

        return deleted_instance_id

    async def _load_balancer_loop(self):
        # this is a simple round-robin load balancer
        round_robin_index = 0
        while True:
            instance_allocation = await self.request_queue.get()
            allocated = False
            logger.info(f"A request is waiting for model {self.model_name}")
            while not allocated:
                # 1. get ready instances
                instance_options = None
                while not instance_options:
                    await asyncio.sleep(1)
                    async with self.instance_management_lock:
                        instance_options = list(self.ready_instances.keys())
                    logger.info(f"{instance_options}")
                logger.info(f"Got ready instances {instance_options}")
                instance_id = instance_options[
                    round_robin_index % len(instance_options)
                ]
                round_robin_index += 1
                async with self.instance_management_lock:
                    if instance_id not in self.ready_instances:
                        continue
                    instance = self.ready_instances[instance_id]
                    allocated = await instance.add_requests(1)
                    if allocated:
                        instance_allocation.set_result(instance_id)

                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            auto_scaling_metrics = {"request_count": self.request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            async with self.instance_management_lock:
                num_running_instances = len(self.starting_instances) + len(
                    self.ready_instances
                )
            logger.info(
                f"Auto-scaler: {num_running_instances} instances, need {desired_instances} instances"
            )
            if desired_instances > num_running_instances:
                logger.info("Creating new instance")
                await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                # logger.info("No scaling needed")
                pass
            await asyncio.sleep(self.loop_interval)

    async def _create_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # TODO: Add max_queue_length to instance
        instance = InstanceHandle(instance_id=instance_id, max_queue_length=10)
        async with self.instance_management_lock:
            self.starting_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))

        return instance_id

    async def _start_instance(self, instance_id):
        async with self.instance_management_lock:
            if instance_id not in self.starting_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        startup_node = (
            await self.model_loading_scheduler.allocate_resource.remote(
                self.model_name, self.resource_requirements
            )
        )
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        await start_instance.options(
            resources={
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            }
        ).remote(
            instance_id,
            self.backend,
            self.model_name,
            self.backend_config,
            startup_config,
        )
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        await instance.backend_instance.init_backend.remote()
        async with self.instance_management_lock:
            self.ready_instances[instance_id] = instance
            self.starting_instances.pop(instance_id)
        return instance_id

    async def _stop_instance(self, instance_id: Optional[str] = None):
        # check if the instance has not started
        while (
            len(self.ready_instances) == 0 and len(self.starting_instances) > 0
        ):
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                instance_id, instance = self.ready_instances.popitem()
            elif instance_id in self.ready_instances:
                instance = self.ready_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_instances[instance_id] = instance
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self.loop.create_task(self._finish_instance(instance_id))

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_instances.pop(instance_id)

        # ensure there's no requests being processed
        while instance.queue_length > 0:
            logger.info(
                f"going to wait, queue length is {instance.queue_length}"
            )
            await asyncio.sleep(1)
        async with instance.lock:
            instance.status = False
        await instance.backend_instance.stop.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            instance.node_id, self.resource_requirements
        )

    async def _shutdown_instance(self, instance_id: str):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if instance_id not in self.ready_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.ready_instances.pop(instance_id)
            async with instance.lock:
                instance.status = False
        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            instance.node_id, self.resource_requirements
        )
        return
