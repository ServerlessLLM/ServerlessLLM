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
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from sllm.kv_store import RedisStore
from sllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class LBConfig:
    max_buffer_size: int = 10
    cold_start_timeout: float = 120.0


@dataclass
class InstanceInfo:
    host: str
    port: int
    instance_id: str


class LoadBalancerService:
    """Per-model load balancer with cold-start buffer and concurrency tracking."""

    def __init__(
        self,
        model: str,
        backend: str,
        store: RedisStore,
        config: LBConfig,
        port: int,
    ):
        self.model = model
        self.backend = backend
        self.model_identifier = f"{model}:{backend}"
        self.store = store
        self.config = config
        self.port = port

        # Cold start buffer
        self.cold_start_buffer: asyncio.Queue = asyncio.Queue(
            maxsize=config.max_buffer_size
        )

        # Concurrency tracking (same as Knative's concurrency metric)
        self.in_flight_count = 0
        self.round_robin_counter = 0

        # HTTP session for forwarding
        self.session: Optional[aiohttp.ClientSession] = None

        self.app = FastAPI(title=f"LB: {model}:{backend}")
        self._setup_routes()

    def _setup_routes(self):
        @self.app.on_event("startup")
        async def startup():
            self.session = aiohttp.ClientSession()
            asyncio.create_task(self._buffer_drain_loop())
            logger.info(
                f"Load Balancer started for {self.model_identifier} on port {self.port}"
            )

        @self.app.on_event("shutdown")
        async def shutdown():
            if self.session:
                await self.session.close()
            logger.info(f"Load Balancer shutdown for {self.model_identifier}")

        @self.app.get("/health")
        async def health():
            return {
                "status": "ok",
                "model": self.model_identifier,
                "buffer_length": self.cold_start_buffer.qsize(),
                "in_flight_count": self.in_flight_count,
            }

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            return await self._handle_request(request, "generate")

        @self.app.post("/v1/completions")
        async def completions(request: Request):
            return await self._handle_request(request, "completions")

        @self.app.post("/v1/embeddings")
        async def embeddings(request: Request):
            return await self._handle_request(request, "encode")

    async def _handle_request(self, request: Request, action: str):
        payload = await request.json()
        instances = await self._get_available_instances()

        if instances:
            # Warm path: forward immediately
            instance = self._select_instance(instances)
            result = await self._forward_to_instance(instance, payload, action)
            return JSONResponse(content=result)

        # Cold path: buffer and wait
        result = await self._buffer_and_wait(payload, action)
        return JSONResponse(content=result)

    async def _get_available_instances(self) -> List[InstanceInfo]:
        """Query worker heartbeats for running instances."""
        workers = await self.store.get_all_workers()
        instances = []
        for worker in workers:
            instances_on_device = worker.get("instances_on_device", {})
            if isinstance(instances_on_device, str):
                try:
                    instances_on_device = json.loads(instances_on_device)
                except (json.JSONDecodeError, TypeError):
                    instances_on_device = {}
            model_instances = instances_on_device.get(self.model_identifier, {})
            if isinstance(model_instances, dict):
                for instance_id, info in model_instances.items():
                    if info.get("status") == "running":
                        instances.append(
                            InstanceInfo(
                                host=worker["node_ip"],
                                port=info["port"],
                                instance_id=instance_id,
                            )
                        )
        return instances

    def _select_instance(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Round-robin selection."""
        idx = self.round_robin_counter % len(instances)
        self.round_robin_counter += 1
        return instances[idx]

    async def _forward_to_instance(
        self, instance: InstanceInfo, payload: dict, action: str
    ) -> dict:
        """Forward request and track concurrency."""
        endpoint_map = {
            "generate": "/v1/chat/completions",
            "completions": "/v1/completions",
            "encode": "/v1/embeddings",
        }
        url = f"http://{instance.host}:{instance.port}{endpoint_map[action]}"

        self.in_flight_count += 1
        await self._update_metrics()

        try:
            async with self.session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                result = await resp.json()
                if resp.status != 200:
                    logger.warning(
                        f"Worker returned non-200 status: {resp.status} for {url}"
                    )
                return result
        except Exception as e:
            logger.error(
                f"Failed to forward request to {instance.instance_id} at {url}: {e}"
            )
            raise HTTPException(
                502, f"Failed to forward request to worker: {str(e)}"
            )
        finally:
            self.in_flight_count -= 1
            await self._update_metrics()

    async def _buffer_and_wait(self, payload: dict, action: str) -> dict:
        """Buffer request during cold start."""
        response_future: asyncio.Future = asyncio.Future()

        try:
            self.cold_start_buffer.put_nowait(
                (payload, action, response_future)
            )
            logger.info(
                f"Buffered request for {self.model_identifier} (buffer size: {self.cold_start_buffer.qsize()})"
            )
        except asyncio.QueueFull:
            logger.warning(
                f"Buffer full for {self.model_identifier}, rejecting request"
            )
            raise HTTPException(503, "Service overloaded, try again later")

        await self._update_metrics()

        try:
            return await asyncio.wait_for(
                response_future, timeout=self.config.cold_start_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"Cold start timeout for {self.model_identifier} after {self.config.cold_start_timeout}s"
            )
            raise HTTPException(
                504, "Cold start timeout - instance did not become available"
            )

    async def _update_metrics(self):
        """Push metrics to Redis for autoscaler."""
        buffer_key = f"lb_buffer:{self.model}:{self.backend}"
        inflight_key = f"lb_inflight:{self.model}:{self.backend}"

        try:
            await self.store.client.set(
                buffer_key, self.cold_start_buffer.qsize(), ex=60
            )
            await self.store.client.set(
                inflight_key, self.in_flight_count, ex=60
            )
        except Exception as e:
            logger.warning(f"Failed to update metrics in Redis: {e}")

    async def _buffer_drain_loop(self):
        """Background: drain buffer when instances become available."""
        while True:
            try:
                if not self.cold_start_buffer.empty():
                    instances = await self._get_available_instances()
                    if instances:
                        try:
                            payload, action, future = (
                                self.cold_start_buffer.get_nowait()
                            )
                            instance = self._select_instance(instances)
                            logger.info(
                                f"Draining buffered request for {self.model_identifier} to {instance.instance_id}"
                            )
                            try:
                                result = await self._forward_to_instance(
                                    instance, payload, action
                                )
                                if not future.done():
                                    future.set_result(result)
                            except Exception as e:
                                if not future.done():
                                    future.set_exception(e)
                        except asyncio.QueueEmpty:
                            pass
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in buffer drain loop: {e}")
                await asyncio.sleep(1)
