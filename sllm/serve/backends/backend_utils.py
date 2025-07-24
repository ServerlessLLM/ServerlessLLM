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
import os
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import aiohttp

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class BackendStatus(Enum):
    UNINITIALIZED = auto()
    RUNNING = auto()
    STOPPING = auto()
    DELETING = auto()


class SllmBackend(ABC):
    @abstractmethod
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    @abstractmethod
    async def init_backend(self) -> None:
        pass

    @abstractmethod
    async def encode(self, request_data: Dict[str, Any]):
        pass

    @abstractmethod
    async def generate(self, request_data: Dict[str, Any]):
        pass

    @abstractmethod
    async def shutdown(self):
        pass

    @abstractmethod
    async def stop(self):
        pass

    @abstractmethod
    async def get_current_tokens(self) -> List[List[int]]:
        pass

    @abstractmethod
    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        pass

    @abstractmethod
    async def fine_tuning(self, request_data: Dict[str, Any]):
        pass


def cleanup_subprocess(process: Optional[subprocess.Popen]) -> None:
    """
    Safely cleanup a subprocess with graceful and force termination.
    
    Args:
        process: The subprocess.Popen object to cleanup
    """
    if process:
        try:
            # Kill the process group to ensure all child processes are terminated
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't shut down gracefully
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()

        except (ProcessLookupError, OSError):
            # Process already terminated
            pass
