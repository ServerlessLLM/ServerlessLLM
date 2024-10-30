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

from sllm.serve.logger import init_logger

# from sllm.serve.utils import AllocationPlan, MigrationPlan
from sllm.serve.routers import RoundRobinRouter
from sllm.serve.schedulers import FcfsScheduler, StorageAwareScheduler
from sllm.serve.store_manager import StoreManager


class SllmControllerException(Exception):
    def __init__(self, message, method):
        real_message = f"[{method}]: {message}"
        super().__init__(real_message)


logger = init_logger(__name__)


# @ray.remote(num_cpus=1, resources={"control_node": 0.1})
class SllmController:
    def __init__(self, config: Optional[Mapping] = None):
        self.config = config

        self.running_lock = asyncio.Lock()
        self.running = False

        self.metadata_lock = asyncio.Lock()
        self.request_routers = {}
        # Register model info
        self.registered_models = {}

    async def start(self):
        async with self.running_lock:
            if self.running:
                logger.error("Controller already started")
                raise RuntimeError("Controller already started")
            self.running = True

        logger.info("Starting store manager")
        enable_storage_aware = self.config.get("enable_storage_aware", False)
        ray_manager_cls = ray.remote(StoreManager)
        self.store_manager = ray_manager_cls.options(
            name="store_manager"
        ).remote()
        await self.store_manager.initialize_cluster.remote()

        logger.info("Starting scheduler")
        if enable_storage_aware:
            ray_scheduler_cls = ray.remote(StorageAwareScheduler)
        else:
            ray_scheduler_cls = ray.remote(FcfsScheduler)

        self.scheduler = ray_scheduler_cls.options(
            name="model_loading_scheduler"
        ).remote()

        self.scheduler.start.remote()

    async def register(self, model_config):
        if not self.running:
            logger.error("Controller not running")
            return
        model_name = model_config.get("model")
        backend = model_config.get("backend", None)
        if backend is None:
            logger.error(f"Backend not specified for model {model_name}")
            return
        backend_config = model_config.get("backend_config", {})
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        async with self.metadata_lock:
            if model_name in self.registered_models:
                logger.error(f"Model {model_name} already registered")
                return

        logger.info(f"Registering new model {model_name}")
        try:
            await self.store_manager.register.remote(model_config)
        except RuntimeError as e:
            error_message = e.args[0]
            raise RuntimeError(f"{error_message}")
        # TODO: put resource requirements in model_config
        resource_requirements = {
            "num_cpus": 1,
            "num_gpus": model_config.get("num_gpus", 0),
        }
        request_router = RoundRobinRouter.options(  # type:ignore
            name=model_name, namespace="models"
        ).remote(model_name, resource_requirements, backend, backend_config)

        async with self.metadata_lock:
            if model_name in self.request_routers:
                logger.error(f"Model {model_name} already registered")
                return

        request_router.start.remote(auto_scaling_config)

        logger.info(f"Model {model_name} registered")

        # Mark model as registered only after model registered successfully
        async with self.metadata_lock:
            self.registered_models[model_name] = model_config
            self.request_routers[model_name] = request_router

    async def update(self, model_name: str, model_config: Mapping):
        async with self.metadata_lock:
            if (
                model_name not in self.registered_models
                or model_name not in self.request_routers
            ):
                logger.error(f"Model {model_name} not found")
                raise ValueError(
                    f"Model {model_name} not found, please register first"
                )

        # update auto-scaling config
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        logger.info(f"Try to update the model {model_name} config")
        if auto_scaling_config is not None:
            async with self.metadata_lock:
                self.registered_models[model_name]["auto_scaling_config"] = (
                    auto_scaling_config
                )
                request_router = self.request_routers[model_name]
            await request_router.update.remote(auto_scaling_config)
        # TODO: update other config (if possible)

    async def exists(self, model_name: str):
        async with self.metadata_lock:
            return model_name in self.registered_models

    async def delete(self, model_name: str):
        router = None
        async with self.metadata_lock:
            if model_name not in self.request_routers:
                logger.error(f"Model {model_name} not found")
                return
            router = self.request_routers.pop(model_name)
            self.registered_models.pop(model_name)

        # if router is not None:
        #     deleted_instance_id = await router.shutdown.remote()
        del router

    async def get_models(self):
        async with self.metadata_lock:
            return self.registered_models

    async def shutdown(self):
        # stop the control loop
        async with self.running_lock:
            if not self.running:
                logger.error("Controller not running")
                raise RuntimeError("Controller not running")
            self.running = False

        # delete all models
        async with self.metadata_lock:
            delete_tasks = [
                self._delete(model_name)
                for model_name in self.request_routers.keys()
            ]
            await asyncio.gather(*delete_tasks)
