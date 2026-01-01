import asyncio
import json
import math
import os
from typing import Any, Dict

from sllm.kv_store import RedisStore
from sllm.logger import init_logger

QUEUE_PER_INSTANCE_THRESHOLD = 5
AUTOSCALER_INTERVAL_SECONDS = 3

logger = init_logger(__name__)


class AutoScaler:
    def __init__(self, store: RedisStore):
        self.store = store
        self._shutdown = asyncio.Event()
        self.interval = AUTOSCALER_INTERVAL_SECONDS
        self.queue_threshold = QUEUE_PER_INSTANCE_THRESHOLD
        self.model_idle_times: Dict[str, int] = {}

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
        all_models = await self.store.get_all_raw_models()
        if not all_models:
            logger.debug("No models registered. Skipping scaling check.")
            return

        logger.debug(f"Running scaling check for {len(all_models)} models.")
        tasks = [
            self._calculate_and_set_decision(model) for model in all_models
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _calculate_and_set_decision(
        self, model_data: Dict[str, Any]
    ) -> None:
        model_name = model_data["model"]
        backend = model_data["backend"]
        model_identifier = f"{model_name}:{backend}"
        decision_key = f"scaling_decision:{model_name}:{backend}"

        if model_data.get("status") == "excommunicado":
            current_instances = await self._count_running_instances(
                model_identifier
            )
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
        target_pending_requests = auto_scaling_config.get(
            "target", self.queue_threshold
        )
        keep_alive = auto_scaling_config.get("keep_alive", 0)

        current_instances = await self._count_running_instances(
            model_identifier
        )
        demand = await self.store.get_queue_length(model_name, backend)
        limbo_up, limbo_down = await self.store.get_limbo_counters(
            model_name, backend
        )

        if demand > 0:
            desired = math.ceil(demand / target_pending_requests)
        else:
            desired = 0

        desired = max(min_instances, min(desired, max_instances))
        effective_capacity = current_instances + limbo_up - limbo_down
        instance_delta = desired - effective_capacity

        if instance_delta < 0:
            idle_time = self.model_idle_times.get(model_identifier, 0)
            if idle_time < keep_alive:
                self.model_idle_times[model_identifier] = (
                    idle_time + self.interval
                )
                logger.info(
                    f"Model '{model_identifier}': keeping instances alive "
                    f"(idle_time: {idle_time + self.interval}s, keep_alive: {keep_alive}s)"
                )
                instance_delta = 0
            else:
                self.model_idle_times[model_identifier] = 0
        else:
            self.model_idle_times[model_identifier] = 0

        if instance_delta != 0:
            logger.info(
                f"Model '{model_identifier}': "
                f"current={current_instances}, demand={demand}, "
                f"limbo_up={limbo_up}, limbo_down={limbo_down} -> "
                f"desired={desired}, effective_capacity={effective_capacity}. "
                f"Decision: change by {instance_delta}."
            )
            await self.store.client.set(decision_key, instance_delta, ex=60)

    async def _count_running_instances(self, model_identifier: str) -> int:
        all_workers = await self.store.get_all_workers()
        count = 0
        for worker in all_workers:
            instances_on_device = worker.get("instances_on_device", {})
            if isinstance(instances_on_device, str):
                try:
                    instances_on_device = json.loads(instances_on_device)
                except (json.JSONDecodeError, TypeError):
                    instances_on_device = {}
            count += len(instances_on_device.get(model_identifier, []))
        return count
