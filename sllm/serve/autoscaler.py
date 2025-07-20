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
import math
import os
from typing import Any, Dict

from sllm.serve.kv_store import RedisStore
from sllm.serve.logger import init_logger
from sllm.serve.schema import *

QUEUE_PER_INSTANCE_THRESHOLD = 5
AUTOSCALER_INTERVAL_SECONDS = 10

logger = init_logger(__name__)


class AutoScaler:
    def __init__(self, store: RedisStore):
        self.store = store
        self._shutdown = asyncio.Event()
        self.interval = AUTOSCALER_INTERVAL_SECONDS
        self.queue_threshold = QUEUE_PER_INSTANCE_THRESHOLD

    async def run_scaling_loop(self) -> None:
        logger.info("Starting AutoScaler service loop...")
        while not self._shutdown.is_set():
            try:
                await self._check_and_scale_all_models()
            except asyncio.CancelledError:
                logger.info("Scaling loop cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"An unexpected error occurred in the autoscaler loop: {e}",
                    exc_info=True,
                )

            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self.interval
                )
            except asyncio.TimeoutError:
                pass

    def shutdown(self) -> None:
        logger.info("Shutting down AutoScaler service...")
        self._shutdown.set()

    async def _check_and_scale_all_models(self) -> None:
        all_models = await self.store.get_all_models()
        if not all_models:
            logger.debug("No models registered. Skipping scaling check.")
            return

        logger.info(f"Running scaling check for {len(all_models)} models.")
        tasks = [
            self._calculate_and_set_decision(model.model_dump()) for model in all_models
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _calculate_and_set_decision(
        self, model_data: Dict[str, Any]
    ) -> None:
        model_name = model_data["model_name"]
        backend = model_data["backend"]
        model_identifier = f"{model_name}:{backend}"
        decision_key = f"scaling_decision:{model_name}:{backend}"

        if model_data.get("status") == "excommunicado":
            current_instances = await self._count_running_instances(model_identifier)
            if current_instances > 0:
                logger.info(
                    f"Model '{model_identifier}' is 'excommunicado'. "
                    f"Decision: scale down by {current_instances} instances."
                )
                await self.store.client.set(
                    decision_key, -current_instances, ex=60
                )
            return

        auto_scaling_config = model_data.get("auto_scaling_config", {})
        min_instances = auto_scaling_config.get("min_instances", 0)
        max_instances = auto_scaling_config.get("max_instances", 1)

        current_instances = await self._count_running_instances(model_identifier)
        queue_length = await self.store.get_queue_length(model_name, backend)

        needed = min_instances
        if queue_length > (current_instances * self.queue_threshold):
            scale_up_target = math.ceil(queue_length / self.queue_threshold)
            needed = max(needed, scale_up_target)

        final_needed = max(min_instances, min(needed, max_instances))
        instance_delta = final_needed - current_instances

        logger.info(
            f"Model '{model_identifier}': "
            f"current={current_instances}, queue={queue_length} -> "
            f"needed={final_needed} (min:{min_instances}, max:{max_instances}). "
            f"Decision: change by {instance_delta}."
        )

        if instance_delta != 0:
            await self.store.client.set(decision_key, instance_delta, ex=60)

    async def _count_running_instances(self, model_identifier: str) -> int:
        """Counts the total number of active instances for a given model identifier."""
        all_workers = await self.store.get_all_workers()
        count = 0
        for worker in all_workers:
            instances_on_device = worker.instances_on_device
            count += len(instances_on_device.get(model_identifier, []))
        return count
