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

import uuid
from typing import Dict, Any, Optional
from sllm.serve.logger import init_logger

logger = init_logger(__name__)

class InstanceManager:
    def __init__(self):
        self._running_instances: Dict[str, Dict[str, Any]] = {}

    async def start_instance(self, instance_id: str, model_config: Dict[str, Any]) -> bool:
        pass

    async def stop_instance(self, instance_id: str) -> bool:
        pass

    async def run_inference(self, instance_id: str, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def get_running_instances_info(self) -> Dict[str, Any]:
        info = {}
        for model_identifier, instances in self._running_instances.items():
            info[model_identifier] = list(instances.keys())
        return info
