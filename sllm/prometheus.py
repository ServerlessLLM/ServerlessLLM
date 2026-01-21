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
"""Prometheus metrics and file-based service discovery helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    generate_latest,
)


def _split_deployment_id(deployment_id: str) -> Tuple[str, str]:
    parts = deployment_id.rsplit(":", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return deployment_id, "unknown"


class PrometheusMetrics:
    """Prometheus metrics for Router visibility."""

    def __init__(self) -> None:
        self._requests_total = Counter(
            "sllm_router_requests_total",
            "Total number of router requests.",
            ["deployment_id", "backend"],
        )
        self._instances = Gauge(
            "sllm_router_instances",
            "Number of healthy backend instances.",
            ["deployment_id", "backend"],
        )

    def observe_request(self, deployment_id: str) -> None:
        _, backend = _split_deployment_id(deployment_id)
        self._requests_total.labels(
            deployment_id=deployment_id, backend=backend
        ).inc()

    def set_instance_count(self, deployment_id: str, count: int) -> None:
        _, backend = _split_deployment_id(deployment_id)
        self._instances.labels(
            deployment_id=deployment_id, backend=backend
        ).set(count)

    def set_instance_counts(
        self, endpoints_by_deployment: Dict[str, List[str]]
    ):
        for deployment_id, endpoints in endpoints_by_deployment.items():
            self.set_instance_count(deployment_id, len(endpoints))


_metrics: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Return the singleton PrometheusMetrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = PrometheusMetrics()
    return _metrics


def render_metrics() -> Tuple[bytes, str]:
    """Render the current metrics payload and content type."""
    return generate_latest(), CONTENT_TYPE_LATEST


@dataclass
class PrometheusFileSD:
    """Prometheus file_sd writer for backend endpoint discovery."""

    path: str

    def __post_init__(self) -> None:
        target_path = Path(self.path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    def write_targets(self, endpoints_by_deployment: Dict[str, List[str]]):
        groups = []
        for deployment_id, endpoints in sorted(endpoints_by_deployment.items()):
            if not endpoints:
                continue
            _, backend = _split_deployment_id(deployment_id)
            groups.append(
                {
                    "targets": sorted(endpoints),
                    "labels": {
                        "deployment_id": deployment_id,
                        "backend": backend,
                    },
                }
            )

        payload = json.dumps(groups, indent=2, sort_keys=True)
        target_path = Path(self.path)
        tmp_path = target_path.with_name(f"{target_path.name}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        os.replace(tmp_path, target_path)
