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
"""
Single Global Router for ServerlessLLM v1-beta.

Design principles (from docs/v1-beta-scalable-router-design.md):
- KISS: Single global router, no per-deployment processes
- YAGNI: No caching, no circuit breakers until needed
- Explicit: Router reads SQLite directly, no sync loops
- Stateless: Ephemeral state only; crash loses buffered requests (503)
- Separation: Treat router as separate component for future scaling

The Router:
- Reads endpoints from SQLite on every request (no cache)
- Round-robin load balancing across healthy endpoints
- Cold-start buffering (ephemeral, lost on crash)
- Pushes metrics directly to Autoscaler

Terminology:
- deployment_id: Unique identifier for a deployment (format: "{model_name}:{backend}")
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import aiohttp

from sllm.database import Database
from sllm.logger import init_logger

if TYPE_CHECKING:
    from sllm.autoscaler import AutoScaler

logger = init_logger(__name__)


@dataclass
class RouterConfig:
    """Router configuration."""

    max_buffer_size: int = 10
    cold_start_timeout: float = 120.0
    request_timeout: float = 300.0
    retry_failed_endpoint: bool = True


@dataclass
class BufferedRequest:
    """A request waiting in the cold-start buffer."""

    deployment_id: str
    payload: Dict[str, Any]
    path: str
    endpoint_future: asyncio.Future = field(default_factory=asyncio.Future)


class Router:
    """
    Single global router for all deployments.

    Reads endpoints from SQLite on every request, provides round-robin
    load balancing, cold-start buffering, and pushes metrics to autoscaler.

    Ephemeral state (lost on restart):
    - Round-robin index per deployment
    - Cold-start buffer per deployment
    - In-flight counter per deployment
    """

    def __init__(
        self,
        database: Database,
        config: Optional[RouterConfig] = None,
        autoscaler: Optional["AutoScaler"] = None,
    ):
        """
        Initialize the Router.

        Args:
            database: Database instance for reading endpoints
            config: Router configuration
            autoscaler: Optional autoscaler for metrics push
        """
        self.database = database
        self.config = config or RouterConfig()

        # Ephemeral state (lost on restart)
        self._round_robin_idx: Dict[str, int] = defaultdict(int)
        self._round_robin_indices = self._round_robin_idx  # Alias for tests
        self._buffers: Dict[str, asyncio.Queue[BufferedRequest]] = {}
        self._in_flight: Dict[str, int] = defaultdict(int)

        # Autoscaler reference (set after initialization)
        self._autoscaler: Optional[AutoScaler] = autoscaler

        # HTTP session (created lazily)
        self._session: Optional[aiohttp.ClientSession] = None

        # Background task for buffer draining
        self._drain_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info("Router initialized")

    def set_autoscaler(self, autoscaler: AutoScaler):
        """Set the autoscaler for metrics push."""
        self._autoscaler = autoscaler

    @property
    def autoscaler(self) -> Optional["AutoScaler"]:
        """Get the autoscaler reference."""
        return self._autoscaler

    async def start(self):
        """Start the router background tasks."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._buffer_drain_loop())
        logger.info("Router started")

    async def stop(self):
        """Stop the router and clean up resources."""
        self._shutdown = True

        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None

        if self._session:
            await self._session.close()
            self._session = None

        logger.info("Router stopped")

    # -------------------------------------------------------------------------
    # Request Handling
    # -------------------------------------------------------------------------

    async def handle_request(
        self,
        payload: Dict[str, Any],
        path: str = "/v1/chat/completions",
        deployment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Handle an inference request for a deployment.

        Reads endpoints from SQLite on every request, selects one via
        round-robin, and forwards the request. If no endpoints are
        available, buffers the request for cold-start.

        Args:
            payload: Request payload (JSON body)
            path: API path (e.g., "/v1/chat/completions")
            deployment_id: Deployment identifier (required)

        Returns:
            Response from the backend

        Raises:
            Exception: On timeout or forwarding failure
        """
        if deployment_id is None:
            raise ValueError("deployment_id is required")

        # Ensure session exists
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Read endpoints from SQLite (every request, no cache)
        endpoints = self.database.get_deployment_endpoints(deployment_id)

        if endpoints:
            # Warm path: forward immediately
            endpoint = self._select_next_endpoint(deployment_id, endpoints)
            return await self._forward_to_endpoint(
                deployment_id, endpoint, payload, path
            )
        else:
            # Cold path: buffer and wait
            return await self._buffer_and_wait(deployment_id, payload, path)

    def _select_next_endpoint(
        self, deployment_id: str, endpoints: List[str]
    ) -> str:
        """Select the next endpoint using round-robin."""
        idx = self._round_robin_idx[deployment_id] % len(endpoints)
        self._round_robin_idx[deployment_id] += 1
        return endpoints[idx]

    async def _forward_to_endpoint(
        self,
        deployment_id: str,
        endpoint: str,
        payload: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Forward request to a specific endpoint."""
        url = f"http://{endpoint}{path}"

        # Track in-flight
        self._in_flight[deployment_id] += 1
        self._push_metrics(deployment_id)

        try:
            async with self._session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.request_timeout
                ),
            ) as resp:
                result = await resp.json()

                if resp.status >= 500:
                    logger.warning(
                        f"[{deployment_id}] Endpoint {endpoint} returned "
                        f"status {resp.status}"
                    )
                    # Router does NOT mark endpoint unhealthy
                    # (reconciler owns this based on Pylet heartbeats)

                return result

        except aiohttp.ClientError as e:
            logger.error(f"[{deployment_id}] Request to {endpoint} failed: {e}")

            # Retry on another endpoint if available and configured
            if self.config.retry_failed_endpoint:
                endpoints = self.database.get_deployment_endpoints(
                    deployment_id
                )
                # Filter out the failed endpoint
                other_endpoints = [ep for ep in endpoints if ep != endpoint]
                if other_endpoints:
                    other_endpoint = self._select_next_endpoint(
                        deployment_id, other_endpoints
                    )
                    logger.info(
                        f"[{deployment_id}] Retrying on endpoint "
                        f"{other_endpoint}"
                    )
                    # Decrement in-flight before retry (will be incremented again)
                    self._in_flight[deployment_id] -= 1
                    return await self._forward_to_endpoint(
                        deployment_id, other_endpoint, payload, path
                    )

            raise Exception(f"Failed to forward request: {e}")

        except asyncio.TimeoutError:
            logger.error(f"[{deployment_id}] Request to {endpoint} timed out")
            raise Exception(
                f"Request timeout after {self.config.request_timeout}s"
            )

        finally:
            self._in_flight[deployment_id] -= 1
            self._push_metrics(deployment_id)

    async def _buffer_and_wait(
        self,
        deployment_id: str,
        payload: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Buffer request during cold start and wait for endpoint."""
        # Get or create buffer for deployment
        if deployment_id not in self._buffers:
            self._buffers[deployment_id] = asyncio.Queue(
                maxsize=self.config.max_buffer_size
            )

        buffer = self._buffers[deployment_id]

        # Create request with future to receive endpoint when available
        loop = asyncio.get_event_loop()
        endpoint_future = loop.create_future()
        request = BufferedRequest(
            deployment_id=deployment_id,
            payload=payload,
            path=path,
            endpoint_future=endpoint_future,
        )

        try:
            buffer.put_nowait(request)
            logger.info(
                f"[{deployment_id}] Buffered request "
                f"(buffer size: {buffer.qsize()})"
            )
            self._push_metrics(deployment_id)
        except asyncio.QueueFull:
            logger.warning(f"[{deployment_id}] Buffer full, rejecting request")
            raise Exception("Service overloaded - buffer full")

        # Phase 1: Wait for endpoint to become available (cold start timeout)
        try:
            endpoint = await asyncio.wait_for(
                endpoint_future, timeout=self.config.cold_start_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{deployment_id}] Cold start timeout after "
                f"{self.config.cold_start_timeout}s"
            )
            raise Exception(
                f"Cold start timeout - no instance available after "
                f"{self.config.cold_start_timeout}s"
            )

        # Phase 2: Forward to endpoint (request timeout applies)
        return await self._forward_to_endpoint(
            deployment_id, endpoint, payload, path
        )

    async def _buffer_drain_loop(self):
        """Background task to notify waiters when endpoints become available."""
        logger.debug("Buffer drain loop started")

        while not self._shutdown:
            try:
                # Check all buffers
                for deployment_id, buffer in list(self._buffers.items()):
                    if not buffer.empty():
                        # Read endpoints from SQLite
                        endpoints = self.database.get_deployment_endpoints(
                            deployment_id
                        )
                        if endpoints:
                            try:
                                request = buffer.get_nowait()
                                endpoint = self._select_next_endpoint(
                                    deployment_id, endpoints
                                )
                                logger.info(
                                    f"[{deployment_id}] Endpoint available, "
                                    f"unblocking request to {endpoint}"
                                )
                                # Signal endpoint availability (caller forwards)
                                if not request.endpoint_future.done():
                                    request.endpoint_future.set_result(endpoint)
                            except asyncio.QueueEmpty:
                                pass

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in drain loop: {e}")
                await asyncio.sleep(1)

        logger.debug("Buffer drain loop stopped")

    def _push_metrics(self, deployment_id: str):
        """Push metrics immediately to autoscaler."""
        if self._autoscaler:
            buffer = self._buffers.get(deployment_id)
            buffer_len = buffer.qsize() if buffer else 0
            in_flight = self._in_flight.get(deployment_id, 0)

            self._autoscaler.receive_metrics(
                deployment_id=deployment_id,
                buffer_len=buffer_len,
                in_flight=in_flight,
            )

    # -------------------------------------------------------------------------
    # Metrics Access (for status endpoints)
    # -------------------------------------------------------------------------

    def get_buffer_length(self, deployment_id: str) -> int:
        """Get the buffer length for a deployment."""
        buffer = self._buffers.get(deployment_id)
        return buffer.qsize() if buffer else 0

    def get_in_flight(self, deployment_id: str) -> int:
        """Get the in-flight count for a deployment."""
        return self._in_flight.get(deployment_id, 0)

    def get_total_demand(self, deployment_id: str) -> int:
        """Get total demand (buffer + in-flight) for a deployment."""
        return self.get_buffer_length(deployment_id) + self.get_in_flight(
            deployment_id
        )

    def get_in_flight_count(self, deployment_id: str) -> int:
        """Alias for get_in_flight (for test compatibility)."""
        return self.get_in_flight(deployment_id)

    def _select_endpoint(self, deployment_id: str) -> Optional[str]:
        """Select an endpoint using round-robin (sync version for tests)."""
        endpoints = self.database.get_deployment_endpoints(deployment_id)
        if not endpoints:
            return None
        return self._select_next_endpoint(deployment_id, endpoints)

    def get_endpoint_count(self, deployment_id: str) -> int:
        """Get the number of healthy endpoints for a deployment."""
        endpoints = self.database.get_deployment_endpoints(deployment_id)
        return len(endpoints)

    # -------------------------------------------------------------------------
    # Draining for Shutdown
    # -------------------------------------------------------------------------

    async def drain(self, timeout: float = 30.0):
        """
        Drain all pending requests before shutdown.

        Waits for in-flight requests to complete and buffers to empty.
        """
        logger.info("Draining router...")

        start_time = asyncio.get_event_loop().time()
        all_drained = False
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check all deployments with buffers OR in-flight requests
            all_deployment_ids = set(self._buffers.keys()) | set(
                self._in_flight.keys()
            )

            all_drained = True
            for deployment_id in all_deployment_ids:
                if (
                    self.get_buffer_length(deployment_id) > 0
                    or self.get_in_flight(deployment_id) > 0
                ):
                    all_drained = False
                    break

            if all_drained:
                break

            await asyncio.sleep(0.05)  # Check more frequently

        if not all_drained:
            logger.warning("Router drain timeout")
        else:
            logger.info("Router drain complete")

    def __repr__(self) -> str:
        total_buffer = sum(
            self.get_buffer_length(m) for m in self._buffers.keys()
        )
        total_inflight = sum(self._in_flight.values())
        return f"Router(buffer={total_buffer}, in_flight={total_inflight})"


# Global router instance
_router: Optional[Router] = None


def get_router() -> Optional[Router]:
    """Get the global Router instance."""
    return _router


def init_router(
    database: Database,
    config: Optional[RouterConfig] = None,
) -> Router:
    """Initialize the global Router instance."""
    global _router
    _router = Router(database=database, config=config)
    return _router
