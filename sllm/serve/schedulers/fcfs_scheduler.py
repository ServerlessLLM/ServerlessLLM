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
        self, model_name: str, resource_requirements: Mapping
    ) -> int:
        logger.info(f"Model {model_name} requested")
        # TODO: consider other resources
        num_gpus = resource_requirements.get("num_gpus", 0)
        async with self.queue_lock:
            if model_name not in self.model_loading_queues:
                self.model_loading_queues[model_name] = []
            allocation_result = self.loop.create_future()
            self.model_loading_queues[model_name].append(
                (time.time(), num_gpus, allocation_result)
            )
        logger.info(f"Model {model_name} added to the loading queue")
        return await allocation_result

    async def deallocate_resource(self, node_id: int, resources: Mapping):
        # TODO: consider other resources
        num_gpus = resources.get("num_gpus", 0)
        logger.info(f"Node {node_id} deallocated {num_gpus} GPUs")
        async with self.metadata_lock:
            if node_id not in self.worker_nodes:
                logger.error(f"Node {node_id} not found")
                return
            self.worker_nodes[node_id]["free_gpu"] += num_gpus

    async def _control_loop(self):
        logger.info("Starting control loop")
        loading_requests = []
        try:
            # first work on earlier requests left in the previou queue
            # after loading_requests becomes empty, get new requests
            while self.running:
                if len(loading_requests) == 0:
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
                    loading_requests.sort(key= lambda x: x[1])
                # logger.info(f"Loading requests: {loading_requests}")
                # first come first serve
                else:
                    worker_nodes = await self._get_worker_nodes()
                    logger.info(f"Worker nodes: {worker_nodes}")
                    # loading_requests.sort(key=lambda x: x[1])
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
                            if node_info["free_gpu"] >= float(num_gpus):
                                if allocation_result.done():
                                    keep_going = True
                                    break
                                try:
                                    self.model_loading_queues[model_name].remove((request_time, num_gpus, allocation_result))
                                except ValueError: 
                                    keep_going = True
                                    break
                                async with self.queue_lock:                              
                                    allocation_result.set_result(node_id)
                                allocated = True
                                logger.info(
                                    f"Allocated node {node_id} for model {model_name}"
                                )
                                node_info["free_gpu"] -= float(num_gpus)
                                break
                        if keep_going: # current request is already allocated, check next
                            continue
                        if not allocated:
                            logger.info(f"No available node for model {model_name}")
                    await self._update_worker_nodes(worker_nodes)
                    # remove the allocated requests from the `loading_requests` list
                    loading_requests = [req for req in loading_requests if not req[4].done()] 
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error: {e}",exc_info=True)

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
