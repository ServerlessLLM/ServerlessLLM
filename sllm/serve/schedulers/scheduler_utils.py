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
from abc import ABC, abstractmethod
from typing import Mapping, Optional

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class SllmScheduler(ABC):
    @abstractmethod
    def __init__(self, scheduler_config: Optional[Mapping] = None):
        super().__init__()

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        pass

    @abstractmethod
    async def allocate_resource(
        self, model_name: str, resource_requirements: Mapping
    ):
        pass

    @abstractmethod
    async def deallocate_resource(self, node_id: int, resources: Mapping):
        pass
