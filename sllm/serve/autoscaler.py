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
from typing import Dict, Any

from sllm.serve.kv_store import RedisStore
from sllm.serve.model_manager import ModelManager
from sllm.serve.worker_manager import WorkerManager
from sllm.serve.logger import init_logger

QUEUE_PER_INSTANCE_THRESHOLD = 5
AUTOSCALER_INTERVAL_SECONDS = 10

logger = init_logger(__name__)

class AutoScaler:
    def __init__(
        self,
        store: RedisStore,
        model_manager: ModelManager,
        worker_manager: WorkerManager,
    ):
        self.store = store
        self.model_manager = model_manager
        self.worker_manager = worker_manager
        self.is_shutting_down = False
        self.interval = AUTOSCALER_INTERVAL_SECONDS
        self.queue_threshold = QUEUE_PER_INSTANCE_THRESHOLD

    async def run_scaling_loop(self) -> None:
        logger.info("Starting AutoScaler service loop...")
        while not self.is_shutting_down:
            try:
                await self._check_and_scale_all_models()
            except Exception as e:
                logger.error(f"An unexpected error occurred in the autoscaler loop: {e}", exc_info=True)
            
            await asyncio.sleep(self.interval)

    def shutdown(self) -> None:
        logger.info("Shutting down AutoScaler service...")
        self.is_shutting_down = True

    async def _check_and_scale_all_models(self) -> None:
        all_models = await self.model_manager.get_all_models()
        if not all_models:
            logger.debug("No models registered. Skipping scaling check.")
            return

        logger.info(f"Running scaling check for {len(all_models)} models.")
        tasks = [self._calculate_and_set_needed_instances(model) for model in all_models]
        await asyncio.gather(*tasks)

    async def _calculate_and_set_needed_instances(self, model_data: Dict[str, Any]) -> None:
        model_name = model_data['model_name']
        backend = model_data['backend']
        model_identifier = f"{model_name}:{backend}"
        
        auto_scaling_config = model_data.get('auto_scaling_config', {})
        min_instances = auto_scaling_config.get('min_instances', 1)
        max_instances = auto_scaling_config.get('max_instances', 1)

        current_instances = await self.worker_manager.count_running_instances(model_identifier)
        queue_length = await self.store.get_queue_length(model_name, backend)

        needed = min_instances

        if current_instances > 0:
            if queue_length > (current_instances * self.queue_threshold):
                scale_up_target = math.ceil(queue_length / self.queue_threshold)
                needed = max(needed, scale_up_target)
        elif queue_length > 0:
            scale_up_target = math.ceil(queue_length / self.queue_threshold)
            needed = max(needed, scale_up_target)

        final_needed = max(min_instances, min(needed, max_instances))

        logger.info(
            f"Model '{model_identifier}': "
            f"current={current_instances}, queue={queue_length} -> "
            f"calculated_needed={final_needed} (min:{min_instances}, max:{max_instances})"
        )

        await self.model_manager.set_needed_instances(model_name, backend, final_needed)


async def main():
    store = RedisStore(host=os.getenv("REDIS_HOST", "localhost"))
    model_manager = ModelManager(store)
    worker_manager = WorkerManager(store)
    
    autoscaler = AutoScaler(
        store=store,
        model_manager=model_manager,
        worker_manager=worker_manager
    )

    try:
        await autoscaler.run_scaling_loop()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        autoscaler.shutdown()
        await store.close()

if __name__ == "__main__":
    asyncio.run(main())
