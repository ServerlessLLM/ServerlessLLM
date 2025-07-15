# ---------------------------------------------------------------------------- # 
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
from typing import List, Optional, Dict, Any

from sllm.serve.logger import init_logger
from sllm.serve.kv_store import RedisStore

logger = init_logger(__name__)
ModelConfig = Dict[str, Any]

class ModelManager:
    def __init__(self, store: RedisStore):
        self.store = store

    async def register(self, model_config: ModelConfig) -> None:
        model_name = model_config.get("model")
        backend = model_config.get("backend")
        
        if not model_name or not backend:
            raise ValueError("Model configuration must include 'model' and 'backend' keys.")

        logger.info(f"Registering model '{model_name}:{backend}'")
        await self.store.register_model(model_config)
        logger.info(f"Successfully registered model '{model_name}:{backend}'")

    async def update(
        self,
        model_name: str,
        backend: str,
        model_config: ModelConfig
    ) -> Optional[ModelConfig]:
        logger.info(f"Attempting to update model '{model_name}:{backend}'")
        existing_model = await self.store.get_model(model_name, backend)
        if not existing_model:
            logger.warning(f"Update failed: model '{model_name}:{backend}' not found.")
            return None
        
        # Ensure the keys are not changed during an update
        model_config["model"] = model_name
        model_config["backend"] = backend
        
        await self.store.register_model(model_config)
        logger.info(f"Successfully updated model '{model_name}:{backend}'")
        return model_config

    async def delete(self, model_name: str, backend: str) -> int:
        logger.info(f"Attempting to delete model '{model_name}:{backend}'")
        deleted_count = await self.store.delete_model(model_name, backend)
        if deleted_count > 0:
            logger.info(f"Successfully deleted model '{model_name}:{backend}'")
        else:
            logger.warning(f"Deletion failed: model '{model_name}:{backend}' not found.")
        return deleted_count

    async def set_needed_instances(self, model_name: str, backend: str, count: int) -> None:
        """Updates the 'instances_needed' field for a model."""
        key = self._get_model_key(model_name, backend)
        await self.store.client.hset(key, "instances_needed", count)

    @staticmethod
    def get_queue_name(model_name: str, backend: str) -> str:
        return f"queue:{model_name}:{backend}"

    async def get_queue_depth(self, model_name: str, backend: str) -> int:
        queue_name = self.get_queue_name(model_name, backend)
        logger.debug(f"Fetching queue depth for '{queue_name}'")
        depth = await self.store.get_queue_depth(queue_name)
        logger.debug(f"Queue depth for '{queue_name}' is {depth}")
        return depth

    async def get_model(self, model_name: str, backend: str) -> Optional[ModelConfig]:
        logger.debug(f"Fetching configuration for model '{model_name}:{backend}'")
        return await self.store.get_model(model_name, backend)

    async def get_all_models(self) -> List[ModelConfig]:
        logger.debug("Fetching all model configurations")
        return await self.store.get_all_models()

    async def get_status(self) -> List[Dict[str, Any]]:
        """
        Generates a detailed status for all registered models, including queue depth.
        """
        logger.debug("Generating system status for all models")
        models = await self.get_all_models()
        
        status_list = []
        for model_config in models:
            model_name = model_config.get("model")
            backend = model_config.get("backend")
            if model_name and backend:
                queue_depth = await self.get_queue_depth(model_name, backend)
                status_list.append({
                    "config": model_config,
                    "queue_depth": queue_depth
                })
        return status_list
