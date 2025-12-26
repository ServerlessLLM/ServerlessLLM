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
In-process Load Balancer for ServerlessLLM v1-beta.

Per-model load balancer with:
- Round-robin endpoint selection
- Cold-start request buffering
- Passive health checking (remove endpoint on failure)
- Direct metric access (no Redis)
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

import aiohttp

from sllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class LBConfig:
    """Load balancer configuration."""

    max_buffer_size: int = 10
    cold_start_timeout: float = 120.0
    request_timeout: float = 300.0
    retry_failed_endpoint: bool = True


@dataclass
class BufferedRequest:
    """A request waiting in the cold-start buffer."""

    payload: Dict[str, Any]
    action: str
    path: str
    future: asyncio.Future


class LoadBalancer:
    """
    Per-model in-process load balancer.

    Responsibilities:
    - Round-robin endpoint selection
    - Cold-start request buffering (when no endpoints available)
    - Passive health checking (remove endpoint on request failure)
    - Metrics exposure for autoscaler

    Endpoint management is done externally by the Reconciler via
    add_endpoint() and remove_endpoint() methods.
    """

    def __init__(self, model_id: str, config: Optional[LBConfig] = None):
        """
        Initialize load balancer for a model.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B:vllm")
            config: Load balancer configuration
        """
        self.model_id = model_id
        self.config = config or LBConfig()

        # Endpoint management
        self._endpoints: Set[str] = set()  # Set of "ip:port" strings
        self._endpoints_list: List[str] = []  # For round-robin indexing
        self._endpoints_lock = asyncio.Lock()
        self._round_robin_idx = 0

        # Cold-start buffer
        self._buffer: asyncio.Queue[BufferedRequest] = asyncio.Queue(
            maxsize=self.config.max_buffer_size
        )

        # Concurrency tracking
        self._in_flight_count = 0
        self._in_flight_lock = asyncio.Lock()

        # HTTP session (created lazily)
        self._session: Optional[aiohttp.ClientSession] = None

        # Background tasks
        self._drain_task: Optional[asyncio.Task] = None
        self._shutdown = False

        logger.info(f"LoadBalancer created for {model_id}")

    async def start(self):
        """Start the load balancer background tasks."""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._drain_task is None:
            self._drain_task = asyncio.create_task(self._buffer_drain_loop())
        logger.info(f"LoadBalancer started for {self.model_id}")

    async def stop(self):
        """Stop the load balancer and clean up resources."""
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

        logger.info(f"LoadBalancer stopped for {self.model_id}")

    # -------------------------------------------------------------------------
    # Endpoint Management (called by Reconciler)
    # -------------------------------------------------------------------------

    async def add_endpoint(self, endpoint: str):
        """Add an endpoint to the load balancer."""
        async with self._endpoints_lock:
            if endpoint not in self._endpoints:
                self._endpoints.add(endpoint)
                self._endpoints_list = list(self._endpoints)
                logger.info(
                    f"[{self.model_id}] Added endpoint {endpoint} "
                    f"(total: {len(self._endpoints)})"
                )

    async def remove_endpoint(self, endpoint: str):
        """Remove an endpoint from the load balancer."""
        async with self._endpoints_lock:
            if endpoint in self._endpoints:
                self._endpoints.discard(endpoint)
                self._endpoints_list = list(self._endpoints)
                logger.info(
                    f"[{self.model_id}] Removed endpoint {endpoint} "
                    f"(total: {len(self._endpoints)})"
                )

    def has_endpoint(self, endpoint: str) -> bool:
        """Check if an endpoint is registered."""
        return endpoint in self._endpoints

    @property
    def endpoint_count(self) -> int:
        """Number of registered endpoints."""
        return len(self._endpoints)

    def get_endpoints(self) -> List[str]:
        """Get a copy of all registered endpoints."""
        return list(self._endpoints)

    # -------------------------------------------------------------------------
    # Metrics (accessed by Autoscaler)
    # -------------------------------------------------------------------------

    @property
    def buffer_length(self) -> int:
        """Number of requests waiting in the cold-start buffer."""
        return self._buffer.qsize()

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently being processed."""
        return self._in_flight_count

    @property
    def total_demand(self) -> int:
        """Total demand = buffer + in-flight."""
        return self.buffer_length + self.in_flight_count

    # -------------------------------------------------------------------------
    # Request Handling
    # -------------------------------------------------------------------------

    async def forward(
        self,
        payload: Dict[str, Any],
        path: str = "/v1/chat/completions",
    ) -> Dict[str, Any]:
        """
        Forward a request to an available endpoint.

        If no endpoints are available, the request is buffered until one
        becomes available or the cold-start timeout is reached.

        Args:
            payload: Request payload (JSON body)
            path: API path (e.g., "/v1/chat/completions")

        Returns:
            Response from the backend

        Raises:
            Exception: On timeout or forwarding failure
        """
        # Ensure session exists
        if self._session is None:
            self._session = aiohttp.ClientSession()

        # Try to get an endpoint
        endpoint = await self._select_endpoint()

        if endpoint:
            # Warm path: forward immediately
            return await self._forward_to_endpoint(endpoint, payload, path)

        # Cold path: buffer and wait
        return await self._buffer_and_wait(payload, path)

    async def _select_endpoint(self) -> Optional[str]:
        """Select an endpoint using round-robin."""
        async with self._endpoints_lock:
            if not self._endpoints_list:
                return None

            idx = self._round_robin_idx % len(self._endpoints_list)
            self._round_robin_idx += 1
            return self._endpoints_list[idx]

    async def _forward_to_endpoint(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Forward request to a specific endpoint."""
        url = f"http://{endpoint}{path}"

        async with self._in_flight_lock:
            self._in_flight_count += 1

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
                        f"[{self.model_id}] Endpoint {endpoint} returned "
                        f"status {resp.status}"
                    )
                    # Don't remove endpoint on 5xx - might be temporary

                return result

        except aiohttp.ClientError as e:
            logger.error(f"[{self.model_id}] Request to {endpoint} failed: {e}")
            # Passive health: remove failed endpoint
            await self.remove_endpoint(endpoint)

            # Retry on another endpoint if available
            if self.config.retry_failed_endpoint:
                other_endpoint = await self._select_endpoint()
                if other_endpoint:
                    logger.info(
                        f"[{self.model_id}] Retrying on endpoint {other_endpoint}"
                    )
                    return await self._forward_to_endpoint(
                        other_endpoint, payload, path
                    )

            raise Exception(f"Failed to forward request: {e}")

        except asyncio.TimeoutError:
            logger.error(f"[{self.model_id}] Request to {endpoint} timed out")
            raise Exception(
                f"Request timeout after {self.config.request_timeout}s"
            )

        finally:
            async with self._in_flight_lock:
                self._in_flight_count -= 1

    async def _buffer_and_wait(
        self,
        payload: Dict[str, Any],
        path: str,
    ) -> Dict[str, Any]:
        """Buffer request during cold start and wait for result."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        request = BufferedRequest(
            payload=payload,
            action="forward",
            path=path,
            future=future,
        )

        try:
            self._buffer.put_nowait(request)
            logger.info(
                f"[{self.model_id}] Buffered request "
                f"(buffer size: {self._buffer.qsize()})"
            )
        except asyncio.QueueFull:
            logger.warning(f"[{self.model_id}] Buffer full, rejecting request")
            raise Exception("Service overloaded - buffer full")

        try:
            return await asyncio.wait_for(
                future, timeout=self.config.cold_start_timeout
            )
        except asyncio.TimeoutError:
            logger.error(
                f"[{self.model_id}] Cold start timeout after "
                f"{self.config.cold_start_timeout}s"
            )
            raise Exception(
                f"Cold start timeout - no instance available after "
                f"{self.config.cold_start_timeout}s"
            )

    async def _buffer_drain_loop(self):
        """Background task to drain buffer when endpoints become available."""
        logger.debug(f"[{self.model_id}] Buffer drain loop started")

        while not self._shutdown:
            try:
                if not self._buffer.empty():
                    endpoint = await self._select_endpoint()
                    if endpoint:
                        try:
                            request = self._buffer.get_nowait()
                            logger.info(
                                f"[{self.model_id}] Draining buffered request "
                                f"to {endpoint}"
                            )
                            try:
                                result = await self._forward_to_endpoint(
                                    endpoint, request.payload, request.path
                                )
                                if not request.future.done():
                                    request.future.set_result(result)
                            except Exception as e:
                                if not request.future.done():
                                    request.future.set_exception(e)
                        except asyncio.QueueEmpty:
                            pass

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self.model_id}] Error in drain loop: {e}")
                await asyncio.sleep(1)

        logger.debug(f"[{self.model_id}] Buffer drain loop stopped")

    # -------------------------------------------------------------------------
    # Draining for Shutdown
    # -------------------------------------------------------------------------

    async def drain(self, timeout: float = 30.0):
        """
        Drain all pending requests before shutdown.

        Waits for in-flight requests to complete and buffers to empty.
        """
        logger.info(f"[{self.model_id}] Draining load balancer...")

        start_time = asyncio.get_event_loop().time()
        while (self._in_flight_count > 0 or not self._buffer.empty()) and (
            asyncio.get_event_loop().time() - start_time < timeout
        ):
            await asyncio.sleep(0.5)

        if self._in_flight_count > 0 or not self._buffer.empty():
            logger.warning(
                f"[{self.model_id}] Drain timeout with "
                f"{self._in_flight_count} in-flight, "
                f"{self._buffer.qsize()} buffered"
            )
        else:
            logger.info(f"[{self.model_id}] Drain complete")

    def __repr__(self) -> str:
        return (
            f"LoadBalancer(model_id={self.model_id!r}, "
            f"endpoints={len(self._endpoints)}, "
            f"buffer={self.buffer_length}, "
            f"in_flight={self.in_flight_count})"
        )
