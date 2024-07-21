import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Mapping, Optional

from serverless_llm_store.client import SllmStoreClient

from serverless_llm.serve.logger import init_logger
from serverless_llm.serve.model_downloader import (
    VllmModelDownloader,
    download_transformers_model,
)
from serverless_llm.serve.utils import get_worker_nodes

logger = init_logger(__name__)


@dataclass
class SllmLocalStore:
    node_id: str
    client: SllmStoreClient
    hardware_info: Mapping = field(default_factory=dict)
    disk_models: Mapping = field(default_factory=dict)
    io_queue: List = field(default_factory=list)
    queued_models: Mapping = field(default_factory=dict)
    pinned_memory_pool: Mapping = field(default_factory=dict)

    lock = asyncio.Lock()

    async def register_model(self, model_name: str, pretrained_model_name: str):
        async with self.lock:
            if model_name in self.disk_models:
                logger.error(f"Model {model_name} already registered")
                return
            model_size = self.client.register_model(pretrained_model_name)
            self.disk_models[model_name] = model_size
            logger.info(f"Model {model_name} registered, {self.disk_models}")

        return model_size

    async def get_store_info(self):
        async with self.lock:
            delta_time = 0
            if len(self.io_queue) > 0:
                delta_time = self.io_queue[-1]["estimated_time"] - time.time()
                if delta_time < 0:
                    delta_time = 0
            return self.disk_models, self.pinned_memory_pool, delta_time

    async def load_to_host(self, model_name: str) -> bool:
        async with self.lock:
            if model_name not in self.disk_models:
                logger.error(
                    f"Model {model_name} not found on node {self.node_id}"
                )
                return False
            if model_name in self.pinned_memory_pool:
                logger.info(
                    f"Model {model_name} already loaded to node {self.node_id}"
                )
                return True
            elif model_name in self.queued_models:
                logger.info(
                    f"Model {model_name} is being loaded to node {self.node_id}"
                )
                return True

            model_size = self.disk_models[model_name]
            start_time = time.time()
            if len(self.io_queue) > 0:
                start_time = self.io_queue[-1]["estimated_time"]
            disk_bandwidth = self.hardware_info.get("disk_bandwidth", 1)
            self.io_queue.append(
                {
                    "model_name": model_name,
                    "estimated_time": start_time + model_size / disk_bandwidth,
                }
            )
            self.queued_models[model_name] = True
            logger.info(
                f"Model {model_name} is being loaded to node {self.node_id}, model size: {model_size}, disk bandwidth: {disk_bandwidth}"
            )
            return True

    async def loading_loop(self):
        while True:
            async with self.lock:
                if len(self.io_queue) == 0:
                    await asyncio.sleep(1)
                    continue
                model_info = self.io_queue[0]

            model_name = model_info["model_name"]
            can_load = await self.lru_eviction(model_name)
            if not can_load:
                logger.warning(
                    f"Model {model_name} cannot be loaded to node {self.node_id}"
                )
                await asyncio.sleep(1)
                continue
            ret = self.client.load_into_cpu(model_name)
            self.io_queue.pop(0)
            if not ret:
                logger.error(f"Failed to load model {model_name}")
                continue
            self.pinned_memory_pool[model_name] = time.time()
            logger.info(f"Model {model_name} loaded to host")

    async def lru_eviction(self, model_name):
        # evict the least recently used models until the model can be loaded
        async with self.lock:
            sorted_models = sorted(
                self.pinned_memory_pool.items(), key=lambda x: x[1]
            )
            memory_usage = sum(
                [self.disk_models[model_name] for model_name in sorted_models]
            )
            logger.info(
                f"Memory usage: {memory_usage}, host size: {self.hardware_info['host_size']},"
                f" model size: {self.disk_models[model_name]}"
            )
            while (
                memory_usage + self.disk_models[model_name]
                > self.hardware_info["host_size"]
                and len(sorted_models) > 0
            ):
                model_name, _ = sorted_models.pop(0)
                if model_name not in self.queued_models:
                    self.client.unload_from_cpu(model_name)
                    memory_usage -= self.disk_models[model_name]
                    self.pinned_memory_pool.pop(model_name)
                    logger.info(f"Model {model_name} evicted")
            return (
                memory_usage + self.disk_models[model_name]
                <= self.hardware_info["host_size"]
            )


# @ray.remote(num_cpus=1, resources={"control_node": 0.1})
class SllmStoreManager:
    def __init__(self, hardware_info: Optional[Mapping] = None):
        logger.info("Initializing store manager")
        self.hardware_info = hardware_info

        self.metadata_lock = asyncio.Lock()
        # Storage info
        self.round_robin_index = 0
        self.local_servers = {}
        self.model_info = {}
        self.model_storage_info = {}

    async def initialize_cluster(self) -> bool:
        if not self.hardware_info:
            logger.warning(
                "Hardware info not provided. Storage-aware scheduling might not work properly"
            )
            return False

        uninitialized_nodes = list(self.hardware_info.keys())
        while len(uninitialized_nodes) > 0:
            worker_node_info = get_worker_nodes()
            for node_id in uninitialized_nodes:
                if node_id in worker_node_info:
                    node_address = worker_node_info[node_id]["address"]
                    # TODO: call ping to check if the node is ready
                    self.local_servers[node_id] = SllmLocalStore(
                        node_id, SllmStoreClient(f"{node_address}:8073")
                    )
                    self.local_servers[
                        node_id
                    ].hardware_info = self.hardware_info[node_id]
                    uninitialized_nodes.remove(node_id)
                    logger.info(f"Node {node_id} initialized")
                    break
            logger.info(
                f"Waiting for nodes {uninitialized_nodes} to be initialized"
            )
            await asyncio.sleep(1)

        return True

    async def get_hardware_info(self):
        return self.hardware_info

    async def get_model_info(self, model_name: Optional[str] = None):
        logger.info(f"Getting model info for {model_name}")
        async with self.metadata_lock:
            if model_name is not None:
                return self.model_info.get(model_name, {})
            else:
                return self.model_info

    async def get_store_info(self, node_id: Optional[str] = None):
        async with self.metadata_lock:
            if node_id is not None:
                if node_id not in self.local_servers:
                    logger.error(f"Node {node_id} not found")
                    return {}
                return await self.local_servers[node_id].get_store_info()
            else:
                node_info = {}
                for node_id in self.local_servers:
                    node_info[node_id] = await self.local_servers[
                        node_id
                    ].get_store_info()
                return node_info

    async def load_to_host(self, node_id: str, model_name: str) -> bool:
        async with self.metadata_lock:
            if node_id not in self.local_servers:
                logger.error(f"Node {node_id} not found")
                return False
            local_server = self.local_servers[node_id]
        logger.info(f"Loading model {model_name} to node {node_id}")
        return await local_server.load_to_host(model_name)

    async def register(self, model_config):
        model_name = model_config.get("model")
        backend = model_config.get("backend", None)
        if backend is None:
            logger.error(f"Backend not specified for model {model_name}")
            return
        backend_config = model_config.get("backend_config", {})
        placement_config = model_config.get("placement_config", {})

        if model_name not in self.model_info:
            self.model_storage_info[model_name] = {}
            logger.info(f"Registering new model {model_name}")

            backend = model_config.get("backend", None)
            pretrained_model_name = backend_config.get(
                "pretrained_model_name_or_path", None
            )
            # 1. download this model to one worker using round-robin
            worker_node_info = get_worker_nodes()

            n_nodes = len(worker_node_info)
            assert n_nodes > 0, "No worker nodes found"

            for node_id, node_info in worker_node_info.items():
                node_address = node_info["address"]
                if node_id not in self.local_servers:
                    self.local_servers[node_id] = SllmLocalStore(
                        node_id, SllmStoreClient(f"{node_address}:8073")
                    )

            target_nodes = []
            if placement_config and "target_nodes" in placement_config:
                target_nodes = placement_config["target_nodes"]
                if not all(
                    [node_id in worker_node_info for node_id in target_nodes]
                ):
                    logger.error(
                        f"Invalid target nodes {target_nodes}, worker nodes: {worker_node_info}"
                    )
                    return
            else:
                # round-robin
                node_id = list(worker_node_info.keys())[
                    self.round_robin_index % n_nodes
                ]
                self.round_robin_index += 1
                target_nodes = [node_id]

            logger.info(
                f"Downloading model {pretrained_model_name} to nodes {target_nodes}"
            )
            for node_id in target_nodes:
                if backend == "transformers":
                    # TODO: remove after fix model path problems
                    logger.warning(
                        f"Due to format different issue, please check model path for transformer backend is different compared with using vLLM backend"
                    )
                    await self.download_transformers_model(
                        pretrained_model_name, node_id
                    )
                elif backend == "vllm":
                    await self.download_vllm_model(
                        pretrained_model_name,
                        node_id,
                        model_config.get("num_gpus", 1),
                        backend_config.get("tensor_parallel_size", 1),
                    )
                else:
                    logger.error(f"Backend {backend} not supported")
                    break
                local_server = self.local_servers[node_id]
                model_size = await local_server.register_model(model_name, pretrained_model_name)
                # record the storage info
                self.model_storage_info[model_name][node_id] = True
                logger.info(f"Model {model_name} downloaded to node {node_id}")
            self.model_info[model_name] = model_size
            logger.info(f"Model {model_name} registered")
        else:
            # TOOD: apply new placement config, if given
            pass

    async def download_transformers_model(
        self, pretrained_model_name, node_id
    ) -> int:
        logger.info(
            f"Downloading model {pretrained_model_name} to node {node_id}"
        )
        return await download_transformers_model.options(
            resources={f"worker_node": 0.1, f"worker_id_{node_id}": 0.1}
        ).remote(pretrained_model_name, "float16")

    async def download_vllm_model(
        self, pretrained_model_name, node_id, num_gpus, tensor_parallel_size
    ):
        logger.info(
            f"Downloading model {pretrained_model_name} to node {node_id}"
        )
        vllm_backend_downloader = VllmModelDownloader.options(
            num_gpus=num_gpus,
            resources={f"worker_node": 0.1, f"worker_id_{node_id}": 0.1},
        ).remote()
        return await vllm_backend_downloader.download_vllm_model.remote(
            pretrained_model_name,
            "float16",  # FIXME: use backend_config
            tensor_parallel_size,
        )
