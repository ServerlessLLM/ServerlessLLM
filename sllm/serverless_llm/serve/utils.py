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
from dataclasses import dataclass
from typing import List, Optional

import ray


@dataclass
class MigrationPlan:
    migration_time: float
    target_model: str
    source_node_id: int
    source_instance_id: int
    target_node_id: int


@dataclass
class AllocationPlan:
    node_id: int
    latency: float
    wait_time: float = 0
    migration_plans: Optional[List[MigrationPlan]] = None


def get_worker_nodes():
    ray_nodes = ray.nodes()
    worker_node_info = {}
    for node in ray_nodes:
        ray_node_id = node.get("NodeID", None)
        assert ray_node_id is not None, "NodeID not found"
        resources = node.get("Resources", {})
        assert resources != {}, "Resources not found"
        node_address = node.get("NodeManagerAddress", None)
        assert (
            node_address is not None and node_address != ""
        ), "NodeManagerAddress not found"
        if resources.get("control_node", 0) > 0:
            continue  # Skip the control node

        for key, value in resources.items():
            if key.startswith("worker_id_"):
                node_id = key.split("_")[-1]
                worker_node_info[node_id] = {
                    "ray_node_id": ray_node_id,
                    "address": node_address,
                    "free_gpu": resources.get("GPU", 0),
                    "total_gpu": resources.get("GPU", 0),
                }

    return worker_node_info
