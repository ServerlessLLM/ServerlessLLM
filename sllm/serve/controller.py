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
import datetime
import os
import uuid
from typing import List, Mapping, Optional

import ray

from sllm.serve.fine_tuning_job_store import FineTuningJobStore
from sllm.serve.logger import init_logger
from sllm.serve.routers import MigrationRouter, RoundRobinRouter
from sllm.serve.schedulers import FcfsScheduler, StorageAwareScheduler
from sllm.serve.store_manager import StoreManager


class SllmControllerException(Exception):
    def __init__(self, message, method):
        real_message = f"[{method}]: {message}"
        super().__init__(real_message)


logger = init_logger(__name__)


class SllmController:
    def __init__(self, config: Optional[Mapping] = None):
        self.config = config

        self.running_lock = asyncio.Lock()
        self.running = False

        self.metadata_lock = asyncio.Lock()
        self.request_routers = {}
        # Register model info
        self.registered_models = {}

        self.fine_tuning_jobs = {}
        self.ft_job_lock = asyncio.Lock()
        self.job_store = FineTuningJobStore.remote()

    async def start(self):
        async with self.running_lock:
            if self.running:
                logger.error("Controller already started")
                raise RuntimeError("Controller already started")
            self.running = True

        # Start the fine-tuning job processor
        asyncio.create_task(self._process_ft_jobs())

        logger.info("Starting store manager")
        enable_storage_aware = self.config.get("enable_storage_aware", False)
        ray_manager_cls = ray.remote(StoreManager)
        self.store_manager = ray_manager_cls.options(
            name="store_manager",
            resources={"control_node": 0.1},
        ).remote()
        await self.store_manager.initialize_cluster.remote()

        logger.info("Starting scheduler")
        if enable_storage_aware:
            ray_scheduler_cls = ray.remote(StorageAwareScheduler)
        else:
            ray_scheduler_cls = ray.remote(FcfsScheduler)

        enable_migration = self.config.get("enable_migration", False)
        if enable_migration:
            self.router_cls = ray.remote(MigrationRouter)
        else:
            self.router_cls = ray.remote(RoundRobinRouter)

        self.scheduler = ray_scheduler_cls.options(
            name="model_loading_scheduler", resources={"control_node": 0.1}
        ).remote(
            scheduler_config={
                "enable_migration": enable_migration,
            }
        )
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
        router_config = model_config.get("router_config", {})
        auto_scaling_config = model_config.get("auto_scaling_config", None)
        enable_lora = backend_config.get("enable_lora", False)
        lora_adapters = backend_config.get("lora_adapters", {})
        async with self.metadata_lock:
            if model_name in self.registered_models:
                logger.info(f"Model {model_name} already registered")
                if lora_adapters is not None:
                    for (
                        lora_adapter_name,
                        lora_adapter_path,
                    ) in lora_adapters.items():
                        await self.store_manager.register_lora_adapter.remote(
                            model_name,
                            lora_adapter_name,
                            lora_adapter_path,
                            backend_config,
                        )
                    request_router = self.request_routers[model_name]
                    await request_router.update.remote(
                        lora_adapters=lora_adapters
                    )
                return

        logger.info(f"Registering new model {model_name}")
        try:
            await self.store_manager.register.remote(model_config)
            for lora_adapter_name, lora_adapter_path in lora_adapters.items():
                await self.store_manager.register_lora_adapter.remote(
                    model_name,
                    lora_adapter_name,
                    lora_adapter_path,
                    backend_config,
                )
        except RuntimeError as e:
            error_message = e.args[0]
            raise RuntimeError(f"{error_message}")
        # TODO: put resource requirements in model_config
        resource_requirements = {
            "num_cpus": 1,
            "num_gpus": model_config.get("num_gpus", 0),
        }
        request_router = self.router_cls.options(
            name=model_name,
            namespace="models",
            num_cpus=1,
            resources={"control_node": 0.1},
        ).remote(
            model_name,
            resource_requirements,
            backend,
            backend_config,
            router_config,
            enable_lora,
            lora_adapters,
        )

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

    async def register_ft_job(self, job_config):
        job_id = str(uuid.uuid4())
        job_info = {
            "config": job_config,
            "status": "pending",
            "created_time": datetime.datetime.now().isoformat(),
            "updated_time": datetime.datetime.now().isoformat(),
            "priority": job_config.get("priority", 0),  # Default priority = 0
        }
        await self.job_store.add_job.remote(job_id, job_info)
        return job_id

    async def _process_ft_jobs(self):
        while True:
            try:
                # TODO: Get available resources, calculate resource availability

                # Get pending jobs with resource requirements
                pending_jobs = await self.job_store.get_pending_jobs.remote()

                # Sort jobs by priority/age
                sorted_jobs = sorted(
                    pending_jobs.items(),
                    key=lambda x: (
                        x[1].get("priority", 0),
                        x[1]["created_time"],
                    ),
                )

                # Process jobs based on available resources
                for job_id, job_info in sorted_jobs:
                    await self._start_ft_job(job_id, job_info)

            except Exception as e:
                logger.error(f"Error in fine-tuning job processor: {str(e)}")

            await asyncio.sleep(30)

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

    async def delete(
        self, model_name: str, lora_adapters: Optional[List[str]] = None
    ):
        router = None
        async with self.metadata_lock:
            if model_name not in self.request_routers:
                logger.error(f"Model {model_name} not found")
                return
            if lora_adapters is not None:
                await self.request_routers[model_name].delete_adapters.remote(
                    lora_adapters
                )
                return
            router = self.request_routers.pop(model_name)
            self.registered_models.pop(model_name)

        # if router is not None:
        #     deleted_instance_id = await router.shutdown.remote()
        del router

    async def get_models(self):
        async with self.metadata_lock:
            return self.registered_models

    async def status(self):
        """
        Returns the status of all registered models in OpenAI-compliant format.
        """
        async with self.metadata_lock:
            models = []
            model_folder = os.getenv("MODEL_FOLDER")
            for model_name, config in self.registered_models.items():
                # Extract or calculate relevant fields
                model_path = config.get("_name_or_path", None)
                created_time = (
                    next(
                        (
                            int(os.path.getctime(os.path.abspath(dirpath)))
                            for dirpath, _, _ in os.walk(model_folder)
                            if dirpath.endswith(model_path)
                        ),
                        None,
                    )
                    if model_path
                    else None
                )

                created_time = config.get("created", None)
                allow_create_engine = config.get("allow_create_engine", None)
                allow_sampling = config.get("allow_sampling", None)
                allow_logprobs = config.get("allow_logprobs", None)
                allow_search_indices = config.get("allow_search_indices", None)
                allow_view = config.get("allow_view", None)
                allow_fine_tuning = config.get("allow_fine_tuning", None)
                organization = config.get("organization", "*")
                group = config.get("group", None)
                is_blocking = config.get("is_blocking", None)

                max_model_len = config.get("max_position_embeddings", None)

                model_permission_id = f"modelperm-{model_name}"
                permission = [
                    {
                        "id": model_permission_id,
                        "object": "model_permission",
                        "created": created_time
                        if created_time is not None
                        else None,
                        "allow_create_engine": allow_create_engine
                        if allow_create_engine is not None
                        else None,
                        "allow_sampling": allow_sampling
                        if allow_sampling is not None
                        else None,
                        "allow_logprobs": allow_logprobs
                        if allow_logprobs is not None
                        else None,
                        "allow_search_indices": allow_search_indices
                        if allow_search_indices is not None
                        else None,
                        "allow_view": allow_view
                        if allow_view is not None
                        else None,
                        "allow_fine_tuning": allow_fine_tuning
                        if allow_fine_tuning is not None
                        else None,
                        "organization": organization
                        if organization is not None
                        else None,
                        "group": group if group is not None else None,
                        "is_blocking": is_blocking
                        if is_blocking is not None
                        else None,
                    }
                ]

                # Build the model metadata entry
                model_metadata = {
                    "id": model_name,
                    "object": "model",
                    "created": created_time
                    if created_time is not None
                    else None,
                    "owned_by": "sllm",
                    "root": model_name,
                    "parent": None,
                    "max_model_len": max_model_len
                    if max_model_len is not None
                    else None,
                    "permission": permission,
                }
                models.append(model_metadata)

            return {"object": "list", "models": models}

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

    async def _start_ft_job(self, job_id, job_info):
        try:
            logger.info(f"[FT Job {job_id}] Updating job status to 'running'.")
            # Update job status
            await self.job_store.update_status.remote(job_id, "running")
            logger.info(f"[FT Job {job_id}] Job status updated to 'running'.")

            logger.info(
                f"[FT Job {job_id}] Creating fine-tuning router with resource limits."
            )
            # Create fine-tuning router with resource limits
            resource_requirements = {
                "num_cpus": 1,
                "num_gpus": job_info["config"].get("num_gpus", 0),
            }
            ft_request_router = self.router_cls.options(
                name=f"ft_{job_id}",
                namespace="fine_tuning",
                num_cpus=1,
                num_gpus=job_info["config"].get("num_gpus", 0),
                resources={"control_node": 0.1},
            ).remote(
                job_info["config"]["model"],
                resource_requirements,
                job_info["config"]["ft_backend"],
                job_info["config"].get("backend_config", {}),
                job_info["config"].get("router_config", {}),
            )
            logger.info(f"[FT Job {job_id}] Fine-tuning router created.")

            # Start the router
            await ft_request_router.start.remote({})

            # Start fine-tuning with timeout
            try:
                logger.info(
                    f"[FT Job {job_id}] Starting fine-tuning with timeout {job_info['config'].get('timeout', 3600)} seconds."
                )
                await asyncio.wait_for(
                    ft_request_router.fine_tuning.remote(job_info["config"]),
                    timeout=job_info["config"].get("timeout", 3600),
                )
                logger.info(
                    f"[FT Job {job_id}] Fine-tuning completed successfully."
                )
                await self.job_store.update_status.remote(job_id, "completed")
                logger.info(
                    f"[FT Job {job_id}] Job status updated to 'completed'."
                )
            except asyncio.TimeoutError:
                logger.error(f"[FT Job {job_id}] Fine-tuning job timed out.")
                await self.job_store.update_status.remote(
                    job_id, "failed", error="Job timed out"
                )
                logger.info(
                    f"[FT Job {job_id}] Job status updated to 'failed' due to timeout."
                )

        except Exception as e:
            logger.error(f"[FT Job {job_id}] Exception occurred: {str(e)}")
            await self.job_store.update_status.remote(
                job_id, "failed", error=str(e)
            )
            logger.info(
                f"[FT Job {job_id}] Job status updated to 'failed' due to exception."
            )
        finally:
            try:
                ft_request_router = ray.get_actor(
                    f"ft_{job_id}", namespace="fine_tuning"
                )
                await ft_request_router.shutdown.remote()
                ray.kill(ft_request_router)
                logger.info(f"[FT Job {job_id}] Fine-tuning router cleaned up")
            except Exception as e:
                logger.warning(
                    f"[FT Job {job_id}] Failed to cleanup fine-tuning router: {str(e)}"
                )

    async def get_ft_job_status(self, job_id):
        return await self.job_store.get_job.remote(job_id)

    async def cancel_ft_job(self, job_id):
        job_info = await self.job_store.get_job.remote(job_id)
        if job_info["status"] == "running":
            ft_request_router = ray.get_actor(
                f"ft_{job_id}", namespace="fine_tuning"
            )
            await ft_request_router.cancel.remote()
            await self.job_store.update_status.remote(job_id, "cancelled")
