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
import copy
import time
from typing import Mapping, Optional

from sllm.serve.logger import init_logger
from sllm.serve.utils import get_worker_nodes

from .scheduler_utils import SllmScheduler

logger = init_logger(__name__)


class FcfsScheduler(SllmScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()
        self.scheduler_config = scheduler_config

        self.queue_lock = asyncio.Lock()
        self.model_loading_queues = {}

        self.metadata_lock = asyncio.Lock()
        self.worker_nodes = {}
        self.model_instance = {}

        self.loop = asyncio.get_running_loop()

        self.running_lock = asyncio.Lock()
        self.running = False

    async def start(self) -> None:
        async with self.running_lock:
            if self.running:
                logger.error("FCFS scheduler already started")
                return
            self.running = True
        logger.info("Starting FCFS scheduler")
        self.loop_task = self.loop.create_task(self._control_loop())

    async def shutdown(self) -> None:
        async with self.running_lock:
            if not self.running:
                logger.error("FCFS scheduler not running")
                return
            self.running = False
        async with self.queue_lock:
            self.model_loading_queues = {}
        if self.loop_task is not None:
            await self.loop_task

    async def allocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ) -> int:
        logger.info(f"Model {model_name} requested")
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        async with self.queue_lock:
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (time.time(), num_gpus, allocation_result)
            )
        logger.info(f"Model {model_name} added to the loading queue")
        node_id = await allocation_result
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                self.model_instance[model_name] = {}
            self.model_instance[model_name][instance_id] = node_id
        return node_id

    async def deallocate_resource(
        self, model_name: str, instance_id: str, resources: Mapping
    ):
        logger.info(f"Deallocating model {model_name} instance {instance_id}")
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        async with self.metadata_lock:
            if model_name not in self.model_instance:
                logger.error(f"Model {model_name} not found")
                return
            if instance_id not in self.model_instance[model_name]:
                logger.error(f"Instance {instance_id} not found")
                return
            node_id = self.model_instance[model_name].pop(instance_id)
            logger.info(f"Node {node_id} deallocated {num_gpus} GPUs")
            if node_id not in self.worker_nodes:
                logger.error(f"Node {node_id} not found")
                return
            self.worker_nodes[node_id]["free_gpu"] += num_gpus
        logger.info(f"Model {model_name} instance {instance_id} deallocated")

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            loading_requests = []
            async with self.queue_lock:
                for (
                    model_name,
                    loading_queue,
                ) in self.model_loading_queues.items():
                    for idx, (
                        request_time,
                        num_gpus,
                        allocation_result,
                    ) in enumerate(loading_queue):
                        loading_requests.append(
                            (
                                model_name,
                                idx,
                                request_time,
                                num_gpus,
                                allocation_result,
                            )
                        )
            # logger.info(f"Loading requests: {loading_requests}")
            # first come first serve
            if len(loading_requests) > 0:
                worker_nodes = await self._get_worker_nodes()
                logger.info(f"Worker nodes: {worker_nodes}")
                loading_requests.sort(key=lambda x: x[1])
                for (
                    model_name,
                    idx,
                    request_time,
                    num_gpus,
                    allocation_result,
                ) in loading_requests:
                    allocated = False
                    keep_going = False
                    for node_id, node_info in worker_nodes.items():
                        if node_info["free_gpu"] >= num_gpus:
                            async with self.queue_lock:
                                # allocation_result was set
                                if allocation_result.done():
                                    keep_going = True
                                    break
                                try:
                                    self.model_loading_queues[
                                        model_name
                                    ].remove(
                                        (
                                            request_time,
                                            num_gpus,
                                            allocation_result,
                                        )
                                    )
                                    allocation_result.set_result(node_id)
                                except ValueError:
                                    keep_going = True
                                    break
                            allocated = True
                            logger.info(
                                f"Allocated node {node_id} for model {model_name}"
                            )
                            node_info["free_gpu"] -= num_gpus
                            break
                    # skip current request if removal fails or allocation_result is already set
                    if keep_going:
                        continue
                    if not allocated:
                        logger.info(f"No available node for model {model_name}")
                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(1)

    async def _get_worker_nodes(self):
        worker_nodes = get_worker_nodes()
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                updated_worker_nodes[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes

        return updated_worker_nodes

    # TODO: implement a dedicated class to manage worker nodes
    async def _update_worker_nodes(self, worker_nodes) -> None:
        async with self.metadata_lock:
            updated_worker_nodes = copy.deepcopy(self.worker_nodes)
        for node_id, node_info in worker_nodes.items():
            if node_id not in updated_worker_nodes:
                logger.error(f"Node {node_id} not found")
                continue
            updated_worker_nodes[node_id] = copy.deepcopy(node_info)
        async with self.metadata_lock:
            self.worker_nodes = updated_worker_nodes
        logger.info(f"Worker nodes updated: {updated_worker_nodes}")
