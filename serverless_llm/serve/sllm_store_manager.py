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
from typing import Mapping, Optional

import ray
from serverless_llm_store.client import SllmStoreClient

from serverless_llm.serve.logger import init_logger
from serverless_llm.serve.model_downloader import (
    VllmModelDownloader,
    download_transformers_model,
)
from serverless_llm.serve.utils import get_worker_nodes

logger = init_logger(__name__)


@ray.remote(num_cpus=1, resources={"control_node": 0.1})
class SllmStoreManager:
    def __init__(self, store_config: Optional[Mapping] = None):
        self.store_config = store_config

        self.metadata_lock = asyncio.Lock()
        # Checckpoint store client
        self.sllm_store_clients = {}
        # Storage info
        self.round_robin_index = 0
        self.node_storage_info = {}
        self.model_storage_info = {}

    async def register(self, model_config):
        model_name = model_config.get("model")
        backend = model_config.get("backend", None)
        if backend is None:
            logger.error(f"Backend not specified for model {model_name}")
            return
        backend_config = model_config.get("backend_config", {})
        placement_config = model_config.get("placement_config", {})

        if model_name not in self.model_storage_info:
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
                if node_id not in self.sllm_store_clients:
                    self.sllm_store_clients[node_id] = SllmStoreClient(
                        f"{node_address}:8073"
                    )
                if node_id not in self.node_storage_info:
                    self.node_storage_info[node_id] = {}

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

            for node_id in target_nodes:
                if backend == "transformers":
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
                # register the model to checkpoint store
                if node_id not in self.sllm_store_clients:
                    logger.error(f"Node {node_id} not found")
                    return
                sllm_store_client = self.sllm_store_clients[node_id]
                sllm_store_client.register_model(pretrained_model_name)
                # record the storage info
                self.node_storage_info[node_id][model_name] = True
                self.model_storage_info[model_name][node_id] = True
                logger.info(f"Model {model_name} downloaded to node {node_id}")
        else:
            # TOOD: apply new placement config, if given
            pass

    async def download_transformers_model(self, pretrained_model_name, node_id):
        await download_transformers_model.options(
            resources={f"worker_node": 0.1, f"worker_id_{node_id}": 0.1}
        ).remote(pretrained_model_name, "float16")

        logger.info(
            f"Downloading model {pretrained_model_name} to node {node_id}"
        )

    async def download_vllm_model(
        self, pretrained_model_name, node_id, num_gpus, tensor_parallel_size
    ):
        vllm_backend_downloader = VllmModelDownloader.options(
            num_gpus=num_gpus,
            resources={f"worker_node": 0.1, f"worker_id_{node_id}": 0.1},
        ).remote()
        await vllm_backend_downloader.download_vllm_model.remote(
            pretrained_model_name,
            "float16",  # FIXME: use backend_config
            tensor_parallel_size,
        )
        logger.info(
            f"Downloading model {pretrained_model_name} to node {node_id}"
        )
