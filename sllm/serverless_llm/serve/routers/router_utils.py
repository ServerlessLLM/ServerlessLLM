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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import ray

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class SllmRouter(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
    ) -> None:
        pass

    @abstractmethod
    async def start(self, auto_scaling_config: Dict[str, int]):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def update(self, auto_scaling_config: Dict[str, int]):
        pass

    @abstractmethod
    async def inference(self, request_data: dict):
        pass


@dataclass
class InstanceHandle:
    instance_id: str
    max_queue_length: int

    node_id: Optional[str] = None
    backend_instance: Optional[ray.actor.ActorHandle] = None
    ready: bool = False
    queue_length: int = 0

    lock: asyncio.Lock = asyncio.Lock()

    async def add_requests(self, num_requests: int = 1):
        async with self.lock:
            if not self.ready:
                return False
            if (
                self.queue_length + num_requests > self.max_queue_length
                or self.queue_length + num_requests < 0
            ):
                return False
            self.queue_length += num_requests
            return True
