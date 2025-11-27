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
from typing import Dict, List, Optional

import ray

from sllm.fine_tuning_instance import start_ft_instance
from sllm.inference_instance import start_instance
from sllm.logger import init_logger

from ..utils import InstanceHandle
from .router_utils import SllmRouter

logger = init_logger(__name__)


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


class RoundRobinRouter(SllmRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
        enable_lora: bool = False,
        lora_adapters: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config
        self.router_config = router_config

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        # Fine-tuning instance pools
        self.starting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.fine_tuning_count = 0
        self.fine_tuning_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.enable_lora = enable_lora
        self.loaded_lora_adapters = lora_adapters
        self.lora_lock = asyncio.Lock()

        self.auto_scaler = None
        logger.info(f"Created new handler for model {self.model_name}")

    async def start(
        self, auto_scaling_config: Dict[str, int], mode: str = "inference"
    ):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        if mode == "inference":
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config
            self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
            self.load_balancer = asyncio.create_task(self._load_balancer_loop())
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
        lora_adapters: Optional[Dict[str, str]] = None,
    ):
        if auto_scaling_config is not None:
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config

        if lora_adapters is not None:
            async with self.lora_lock:
                self.loaded_lora_adapters = lora_adapters

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
        logger.info(f"Enqueued {action} request for model {self.model_name}")

        instance_id = await instance_allocation
        logger.info(f"{request_data}, type: {type(request_data)}")
        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return {"error": "Instance not found"}
            instance = self.ready_inference_instances[instance_id]

        # sanity check
        if self.enable_lora and "lora_adapter_name" in request_data:
            lora_adapter_name = request_data["lora_adapter_name"]
            if lora_adapter_name not in self.loaded_lora_adapters:
                logger.error(f"Lora adapter {lora_adapter_name} not found")
                return {"error": f"Lora adapter {lora_adapter_name} not found"}
            await instance.backend_instance.load_lora_adapter.remote(
                lora_name=lora_adapter_name,
                lora_path=self.loaded_lora_adapters[lora_adapter_name],
            )
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

    async def fine_tuning(self, request_data: dict):
        logger.info(f"Starting fine-tuning for model {self.model_name}")
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.fine_tuning_count_lock:
            self.fine_tuning_count += 1

        try:
            instance_id = await self._create_ft_instance()
        except Exception as e:
            logger.error(f"Failed to create fine-tuning instance: {str(e)}")
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {"error": f"Failed to create fine-tuning instance: {str(e)}"}

        max_wait_time = 300
        wait_time = 0
        while wait_time < max_wait_time:
            async with self.instance_management_lock:
                if instance_id in self.ready_ft_instances:
                    instance = self.ready_ft_instances[instance_id]
                    break
                elif instance_id not in self.starting_ft_instances:
                    logger.error(
                        f"Fine tuning instance {instance_id} not found in starting or ready instances"
                    )
                    async with self.fine_tuning_count_lock:
                        self.fine_tuning_count -= 1
                    return {"error": "Fine tuning instance not found"}
            await asyncio.sleep(0.1)
            wait_time += 0.1
        else:
            logger.error(
                f"Timeout waiting for fine tuning instance {instance_id} to be ready"
            )
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {
                "error": "Timeout waiting for fine tuning instance to be ready"
            }

        try:
            logger.info(f"Calling fine_tuning method on instance {instance_id}")
            result = await instance.backend_instance.fine_tuning.remote(
                request_data=request_data
            )

            logger.info(f"Finished processing fine-tuning {self.model_name}")
            await instance.add_requests(-1)

            await self._shutdown_instance(instance_id, is_ft=True)

            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1

            return result
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            await instance.add_requests(-1)
            await self._shutdown_instance(instance_id, is_ft=True)
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            logger.info(
                f"Fine-tuning failed and cleaned up for model {self.model_name}"
            )
            return {"error": f"Fine-tuning failed: {str(e)}"}

    async def delete_adapters(self, lora_adapters: List[str]):
        async with self.lora_lock:
            for adapter_name in lora_adapters:
                if adapter_name in self.loaded_lora_adapters:
                    del self.loaded_lora_adapters[adapter_name]
        logger.info(
            f"Deleted LoRA adapters {lora_adapters} on model {self.model_name}"
        )

    async def shutdown(self):
        async with self.running_lock:
            self.running = False
        # stop all inference instances
        # return all unfinished requests
        while not self.request_queue.empty():
            request_data, done_event = await self.request_queue.get()
            done_event.set_result({"error": "Instance cancelled"})

        async with self.instance_management_lock:
            deleted_instance_id = list(self.ready_inference_instances.keys())
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
                        instance_options = list(
                            self.ready_inference_instances.keys()
                        )
                    logger.info(f"{instance_options}")
                logger.info(f"Got ready instances {instance_options}")
                instance_id = instance_options[
                    round_robin_index % len(instance_options)
                ]
                round_robin_index += 1
                async with self.instance_management_lock:
                    if instance_id not in self.ready_inference_instances:
                        continue
                    instance = self.ready_inference_instances[instance_id]
                    # check if the request queue reaches max length
                    if await instance.check_request_queue():
                        allocated = await instance.add_requests(1)
                        if allocated:
                            instance_allocation.set_result(instance_id)
                    else:
                        logger.info(
                            f"Instance {instance_id} cannot add another request"
                        )
                if not allocated:
                    await asyncio.sleep(self.loop_interval)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            async with self.instance_management_lock:
                num_running_instances = len(
                    self.starting_inference_instances
                ) + len(self.ready_inference_instances)
            logger.info(
                f"{self.model_name}: {num_running_instances} instances,"
                f"need {desired_instances} instances",
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
        # get max_queue_length from auto_scaling_config
        if self.auto_scaling_config.get("metric", "") == "concurrency":
            max_request_length = self.auto_scaling_config.get("target", 1)
        else:
            max_request_length = 1
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))

        return instance_id

    async def _create_ft_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new FT instance {instance_id} for model {self.model_name}"
        )

        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=1,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_ft_instances[instance_id] = instance
        self.loop.create_task(self._start_ft_instance(instance_id))
        logger.info(f"Created task for starting FT instance {instance_id}")
        return instance_id

    async def _start_instance(self, instance_id):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        startup_node = (
            await self.model_loading_scheduler.allocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
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
            num_cpus=self.resource_requirements["num_cpus"],
            num_gpus=self.resource_requirements["num_gpus"],
            resources={
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
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
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)
        return instance_id

    async def _start_ft_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.starting_ft_instances:
                logger.error(f"FT Instance {instance_id} not found")
                return
            instance = self.starting_ft_instances[instance_id]

        logger.info(
            f"Allocating FT resources for model {self.model_name} on {instance_id}"
        )
        try:
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            logger.debug(
                f"Allocated resources on node {startup_node} for FT instance {instance_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to allocate resources for FT instance {instance_id}: {str(e)}"
            )
            raise

        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }

        try:
            instance.backend_instance = await start_ft_instance.options(
                num_cpus=self.resource_requirements["num_cpus"],
                num_gpus=self.resource_requirements["num_gpus"],
                resources=startup_config["resources"],
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
        except Exception as e:
            logger.error(
                f"Failed to create Ray actor for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        try:
            await instance.backend_instance.init_backend.remote()
        except Exception as e:
            logger.error(
                f"Failed to initialize backend for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise

        async with self.instance_management_lock:
            self.ready_ft_instances[instance_id] = instance
            self.starting_ft_instances.pop(instance_id)
        logger.info(f"Fine-tuning instance {instance_id} is now ready")
        return instance_id

    async def _stop_instance(self, instance_id: Optional[str] = None):
        while len(self.ready_inference_instances) <= 0:
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                instance_id, instance = self.ready_inference_instances.popitem()
            elif instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_inference_instances[instance_id] = instance
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self.loop.create_task(self._finish_instance(instance_id))

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        async with instance.lock:
            instance.status = False
        await instance.backend_instance.stop.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )

    async def _shutdown_instance(self, instance_id: str, is_ft: bool = False):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if is_ft:
                pool = self.ready_ft_instances
            else:
                pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
            async with instance.lock:
                instance.status = False
        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )
        return

    # MoE-CAP Batch Recording Methods
    async def start_batch_recording(self):
        """Start batch recording on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            # Get first available instance (or we could broadcast to all)
            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.start_batch_recording.remote()

    async def stop_batch_recording(self):
        """Stop batch recording on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.stop_batch_recording.remote()

    async def dump_batch_recording(self):
        """Dump batch recording from all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                    "records": [],
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.dump_batch_recording.remote()

    async def batch_recording_status(self):
        """Get batch recording status from all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.batch_recording_status.remote()

    async def clear_batch_recording(self):
        """Clear batch recording on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.clear_batch_recording.remote()

    # Expert Distribution Recording Methods
    async def configure_expert_distribution(
        self,
        recording_mode: str = "per_pass",
        enable_metrics: bool = True,
        buffer_size: int = -1,
    ):
        """Configure expert distribution recording on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.configure_expert_distribution.remote(
            recording_mode=recording_mode,
            enable_metrics=enable_metrics,
            buffer_size=buffer_size,
        )

    async def start_expert_distribution_recording(
        self, recording_mode: str = "per_pass"
    ):
        """Start expert distribution recording on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.start_expert_distribution_recording.remote(
            recording_mode=recording_mode
        )

    async def stop_expert_distribution_recording(self):
        """Stop expert distribution recording on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.stop_expert_distribution_recording.remote()

    async def dump_expert_distribution(self, output_path: str = None):
        """Dump expert distribution data from all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                    "worker_data": [],
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return await instance.backend_instance.dump_expert_distribution.remote(
            output_path=output_path
        )

    async def expert_distribution_status(self):
        """Get expert distribution recording status from all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return (
            await instance.backend_instance.expert_distribution_status.remote()
        )

    async def clear_expert_distribution(self):
        """Clear expert distribution data on all ready inference instances."""
        async with self.instance_management_lock:
            if not self.ready_inference_instances:
                return {
                    "status": "error",
                    "message": "No ready instances available",
                }

            instance_id = next(iter(self.ready_inference_instances))
            instance = self.ready_inference_instances[instance_id]

        return (
            await instance.backend_instance.clear_expert_distribution.remote()
        )
