# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2025                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
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
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class FineTuningBackendStatus(Enum):
    UNINITIALIZED = "uninitialized"
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    DELETING = "deleting"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class SllmFineTuningBackend(ABC):
    @abstractmethod
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        pass

    @abstractmethod
    def init_backend(self) -> None:
        pass

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def fine_tuning(self, request_data: Optional[Dict[str, Any]]):
        pass
