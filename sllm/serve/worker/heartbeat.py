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
from sllm.serve.worker.hardware_utils import get_dynamic_metrics
from sllm.serve.worker.instance_manager import InstanceManager

logger = init_logger(__name__)


async def run_heartbeat_loop(
    instance_manager: InstanceManager,
    head_node_url: str,
    node_id: str,
    node_ip: str,
    static_hardware_info: dict,
    app_state,
    interval_seconds: int = 15,
    worker_port: int = 8001,
):
    # Start with provided node_id (may be None for new workers)
    current_node_id = node_id if node_id else None
    
    if current_node_id:
        logger.info(f"Starting heartbeat loop for reconnecting node {current_node_id}. Reporting to {head_node_url}.")
    else:
        logger.info(f"Starting heartbeat loop for new worker. Will receive node_id via /confirmation endpoint.")

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Check if we received node_id via /confirmation endpoint
                if not current_node_id:
                    stored_node_id = getattr(app_state, 'node_id', None)
                    if stored_node_id:
                        current_node_id = stored_node_id
                        logger.info(f"Using node_id received from confirmation: {current_node_id}")

                dynamic_info = get_dynamic_metrics()

                # Build payload - only include node_id if we have one
                payload = {
                    "node_ip": node_ip,
                    "node_port": worker_port,
                    "instances_on_device": instance_manager.get_running_instances_info(),
                    "hardware_info": {**static_hardware_info, **dynamic_info},
                }
                
                # Only include node_id if we have one (for reconnection or after confirmation)
                if current_node_id:
                    payload["node_id"] = current_node_id

                heartbeat_url = f"{head_node_url}/heartbeat"
                async with session.post(
                    heartbeat_url, json=payload
                ) as response:
                    response.raise_for_status()
                    logger.debug(f"Heartbeat sent successfully for node {current_node_id or 'unregistered'}.")

            except Exception as e:
                logger.error(f"Failed to send heartbeat: {e}")

            await asyncio.sleep(interval_seconds)
