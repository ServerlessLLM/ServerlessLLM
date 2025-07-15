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
import os
import uuid
import aiohttp
from sllm.serve.logger import init_logger
from sllm.serve.worker.instance_manager import InstanceManager

logger = init_logger(__name__)

NODE_ID = f"worker-{uuid.uuid4().hex[:8]}"

async def run_heartbeat_loop(
    instance_manager: InstanceManager,
    head_node_url: str,
    interval_seconds: int = 15
):
    """Periodically sends a heartbeat to the head node's API Gateway."""
    logger.info(f"Starting heartbeat loop for node {NODE_ID}. Reporting to {head_node_url}.")
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:

                heartbeat_url = f"{head_node_url}/heartbeat"
                async with session.post(heartbeat_url, json=payload) as response:
                    response.raise_for_status()
                    logger.debug(f"Heartbeat sent successfully for node {NODE_ID}.")

            except Exception as e:
                logger.error(f"Failed to send heartbeat: {e}")

            await asyncio.sleep(interval_seconds)

def get_hardware_info() -> dict:
    # take the existing code
    pass
