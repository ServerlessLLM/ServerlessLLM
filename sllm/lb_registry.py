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
LoadBalancer Registry for ServerlessLLM v1-beta.

Simple dict-based registry for per-model LoadBalancer instances.
Created on model registration, removed on model deletion.
"""

import asyncio
from typing import Dict, List, Optional

from sllm.load_balancer import LBConfig, LoadBalancer
from sllm.logger import init_logger

logger = init_logger(__name__)


class LoadBalancerRegistry:
    """
    Registry for per-model LoadBalancer instances.

    Thread-safe management of LoadBalancer lifecycle:
    - Created when a model is registered
    - Drained and removed when a model is deleted
    """

    def __init__(self):
        self._load_balancers: Dict[str, LoadBalancer] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        model_id: str,
        config: Optional[LBConfig] = None,
    ) -> LoadBalancer:
        """
        Get or create a LoadBalancer for a model.

        Args:
            model_id: Model identifier (e.g., "meta-llama/Llama-3.1-8B:vllm")
            config: Optional LoadBalancer configuration

        Returns:
            LoadBalancer instance
        """
        async with self._lock:
            if model_id not in self._load_balancers:
                lb = LoadBalancer(model_id, config)
                await lb.start()
                self._load_balancers[model_id] = lb
                logger.info(f"Created LoadBalancer for {model_id}")

            return self._load_balancers[model_id]

    def get(self, model_id: str) -> Optional[LoadBalancer]:
        """
        Get a LoadBalancer for a model.

        Args:
            model_id: Model identifier

        Returns:
            LoadBalancer instance or None if not found
        """
        return self._load_balancers.get(model_id)

    def has(self, model_id: str) -> bool:
        """Check if a LoadBalancer exists for a model."""
        return model_id in self._load_balancers

    async def remove(self, model_id: str) -> bool:
        """
        Remove a LoadBalancer for a model.

        Stops the LoadBalancer without draining.

        Args:
            model_id: Model identifier

        Returns:
            True if removed, False if not found
        """
        async with self._lock:
            if model_id in self._load_balancers:
                lb = self._load_balancers.pop(model_id)
                await lb.stop()
                logger.info(f"Removed LoadBalancer for {model_id}")
                return True
            return False

    async def drain_and_remove(
        self,
        model_id: str,
        drain_timeout: float = 30.0,
    ) -> bool:
        """
        Drain and remove a LoadBalancer for a model.

        Waits for pending requests to complete before stopping.

        Args:
            model_id: Model identifier
            drain_timeout: Timeout for draining (seconds)

        Returns:
            True if removed, False if not found
        """
        lb = None
        async with self._lock:
            if model_id in self._load_balancers:
                lb = self._load_balancers.pop(model_id)

        if lb:
            await lb.drain(timeout=drain_timeout)
            await lb.stop()
            logger.info(f"Drained and removed LoadBalancer for {model_id}")
            return True

        return False

    def list_models(self) -> List[str]:
        """Get list of models with active LoadBalancers."""
        return list(self._load_balancers.keys())

    def get_all(self) -> Dict[str, LoadBalancer]:
        """Get a copy of all LoadBalancers."""
        return dict(self._load_balancers)

    @property
    def count(self) -> int:
        """Number of active LoadBalancers."""
        return len(self._load_balancers)

    async def shutdown(self):
        """
        Shutdown all LoadBalancers.

        Drains and stops all LoadBalancers in parallel.
        """
        logger.info(f"Shutting down {len(self._load_balancers)} LoadBalancers")

        async with self._lock:
            tasks = []
            for model_id, lb in list(self._load_balancers.items()):

                async def shutdown_lb(m_id, l):
                    await l.drain(timeout=10.0)
                    await l.stop()
                    logger.info(f"Shutdown LoadBalancer for {m_id}")

                tasks.append(shutdown_lb(model_id, lb))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            self._load_balancers.clear()

        logger.info("All LoadBalancers shutdown complete")

    def __repr__(self) -> str:
        return f"LoadBalancerRegistry(count={len(self._load_balancers)})"


# Global registry instance
_registry: Optional[LoadBalancerRegistry] = None


def get_lb_registry() -> LoadBalancerRegistry:
    """Get the global LoadBalancer registry instance."""
    global _registry
    if _registry is None:
        _registry = LoadBalancerRegistry()
    return _registry


def init_lb_registry() -> LoadBalancerRegistry:
    """Initialize the global LoadBalancer registry."""
    global _registry
    _registry = LoadBalancerRegistry()
    return _registry
