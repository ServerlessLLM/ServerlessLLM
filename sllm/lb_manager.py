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
import socket
from typing import Dict, Optional

import uvicorn

from sllm.kv_store import RedisStore
from sllm.load_balancer import LBConfig, LoadBalancerService
from sllm.logger import init_logger

logger = init_logger(__name__)

# Port range for load balancers
LB_PORT_START = 9000
LB_PORT_END = 9999


class LoadBalancerManager:
    """Manages lifecycle of per-model Load Balancer processes."""

    def __init__(self, store: RedisStore):
        self.store = store
        self.running_lbs: Dict[str, asyncio.Task] = {}
        self.lb_services: Dict[str, LoadBalancerService] = {}
        self._allocated_ports: set = set()

    def _allocate_port(self) -> int:
        """Allocate an available port for a load balancer."""
        for port in range(LB_PORT_START, LB_PORT_END):
            if port not in self._allocated_ports:
                # Try to bind to check if port is actually available
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", port))
                        self._allocated_ports.add(port)
                        return port
                except OSError:
                    continue
        raise RuntimeError("No available ports for load balancer")

    def _get_host_ip(self) -> str:
        """Get the host IP address."""
        try:
            # Get the IP that would be used to connect to an external host
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    async def start_lb(
        self, model: str, backend: str, lb_config: dict = None
    ) -> str:
        """Start LB for model, register endpoint."""
        model_identifier = f"{model}:{backend}"

        # Check if LB already running
        if model_identifier in self.running_lbs:
            logger.warning(
                f"Load balancer for {model_identifier} already running"
            )
            return await self.get_lb_endpoint(model, backend)

        config = LBConfig(**(lb_config or {}))
        port = self._allocate_port()

        logger.info(
            f"Starting load balancer for {model_identifier} on port {port}"
        )

        lb = LoadBalancerService(model, backend, self.store, config, port)
        self.lb_services[model_identifier] = lb

        # Run uvicorn in background task
        uvicorn_config = uvicorn.Config(
            lb.app, host="0.0.0.0", port=port, log_level="warning"
        )
        server = uvicorn.Server(uvicorn_config)
        task = asyncio.create_task(server.serve())

        self.running_lbs[model_identifier] = task

        # Register endpoint
        endpoint = f"{self._get_host_ip()}:{port}"
        await self.store.client.set(
            f"lb_endpoint:{model}:{backend}", endpoint, ex=None
        )

        logger.info(
            f"Load balancer for {model_identifier} started at {endpoint}"
        )
        return endpoint

    async def stop_lb(self, model: str, backend: str):
        """Stop LB when model deleted."""
        model_identifier = f"{model}:{backend}"

        if model_identifier not in self.running_lbs:
            logger.warning(
                f"Load balancer for {model_identifier} not running, nothing to stop"
            )
            return

        logger.info(f"Stopping load balancer for {model_identifier}")

        # Cancel the task
        task = self.running_lbs[model_identifier]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Cleanup
        del self.running_lbs[model_identifier]
        if model_identifier in self.lb_services:
            lb_service = self.lb_services[model_identifier]
            if lb_service.port in self._allocated_ports:
                self._allocated_ports.remove(lb_service.port)
            del self.lb_services[model_identifier]

        # Cleanup Redis keys
        await self.store.client.delete(f"lb_endpoint:{model}:{backend}")
        await self.store.client.delete(f"lb_buffer:{model}:{backend}")
        await self.store.client.delete(f"lb_inflight:{model}:{backend}")

        logger.info(f"Load balancer for {model_identifier} stopped")

    async def get_lb_endpoint(self, model: str, backend: str) -> Optional[str]:
        """Get LB endpoint for routing."""
        endpoint = await self.store.client.get(f"lb_endpoint:{model}:{backend}")
        return endpoint.decode() if endpoint else None

    async def shutdown_all(self):
        """Shutdown all running load balancers."""
        logger.info("Shutting down all load balancers...")
        tasks = []
        for model_identifier in list(self.running_lbs.keys()):
            model, backend = model_identifier.split(":", 1)
            tasks.append(self.stop_lb(model, backend))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All load balancers shut down")
