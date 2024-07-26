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
from typing import Mapping, Optional

import ray

from serverless_llm.serve.logger import init_logger

from .fcfs_scheduler import FcfsScheduler

logger = init_logger(__name__)


# @ray.remote(num_cpus=1, resources={"control_node": 0.1})
class StorageAwareScheduler(FcfsScheduler):
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__(scheduler_config)

        self.store_manager = None

    async def _control_loop(self):
        logger.info("Starting control loop")
        while self.running:
            loading_requests = []
            logger.info(f"Loading requests: {loading_requests}")
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
            logger.info(f"Loading requests are: {loading_requests}")
            if self.store_manager is None:
                try:
                    self.store_manager = ray.get_actor("sllm_store_manager")
                except ValueError:
                    logger.error("Store manager not found")
                    await asyncio.sleep(1)
                    continue
            # first come first serve
            if len(loading_requests) > 0:
                worker_nodes = await self._get_worker_nodes()
                logger.info(f"Worker nodes: {worker_nodes}")
                model_info = await self.store_manager.get_model_info.remote()
                logger.info(f"Model info: {model_info}")
                store_info = await self.store_manager.get_store_info.remote()
                logger.info(f"Store info: {store_info}")
                hardware_info = (
                    await self.store_manager.get_hardware_info.remote()
                )
                logger.info(f"Hardware info: {hardware_info}")
                loading_requests.sort(key=lambda x: x[1])
                logger.info(f"Sorted loading requests: {loading_requests}")
                for (
                    model_name,
                    idx,
                    request_time,
                    num_gpus,
                    allocation_result,
                ) in loading_requests:
                    logger.info(f"Processing request for model {model_name}")
                    scheduling_options = []
                    logger.info(f"Checking model {model_name}")
                    for node_id, node_info in worker_nodes.items():
                        logger.info(
                            f"Checking node {node_id}, node info: {node_info}"
                        )
                        free_gpu = node_info["free_gpu"]
                        logger.info(f"Node {node_id} has {free_gpu} free GPUs")
                        if free_gpu < num_gpus:
                            continue
                        if node_id not in store_info:
                            logger.error(
                                f"Node {node_id} not found in store info"
                            )
                            continue
                        latency = 0
                        (
                            node_store_info,
                            pinned_memory_pool,
                            node_waiting_time,
                        ) = store_info[node_id]
                        if model_name not in node_store_info:
                            logger.info(
                                f"Model {model_name} not found in node {node_id}"
                            )
                            # Note(Yao): Downloading from HuggingFace Hub is
                            # slower than network bandwidth and difficult to estimate.
                            # So we just consider local checkpoints for now.
                            continue
                        if model_name not in pinned_memory_pool:
                            latency += (
                                node_waiting_time
                                + model_info[model_name]
                                / hardware_info[node_id]["disk_bandwidth"]
                            )
                        else:
                            latency += (
                                model_info[model_name]
                                / hardware_info[node_id]["pcie_bandwidth"]
                            )
                        scheduling_options.append((node_id, latency))

                    # sort by latency
                    if scheduling_options:
                        scheduling_options.sort(key=lambda x: x[1])
                        logger.info(
                            f"Sorted scheduling options: {scheduling_options}"
                        )
                        node_id, _ = scheduling_options[0]
                        async with self.queue_lock:
                            self.model_loading_queues[model_name].pop(idx)
                            allocation_result.set_result(node_id)
                        logger.info(
                            f"Allocated node {node_id} for model {model_name}"
                        )
                        node_info = worker_nodes[node_id]
                        node_info["free_gpu"] -= num_gpus
                        await self.store_manager.load_to_host.remote(
                            node_id, model_name
                        )
                    else:
                        logger.info(f"No available node for model {model_name}")
                await self._update_worker_nodes(worker_nodes)

            await asyncio.sleep(1)
