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
import json
import re

import pytest
from fastapi.testclient import TestClient

from sllm.api_gateway import create_app
from sllm.prometheus import PrometheusFileSD, get_metrics, render_metrics


def _metric_value(payload: str, name: str, labels: dict) -> float:
    for line in payload.splitlines():
        if not line.startswith(f"{name}{{"):
            continue
        label_part, value_part = line.split("}", 1)
        label_str = label_part.split("{", 1)[1]
        parsed = {}
        for entry in label_str.split(","):
            key, value = entry.split("=", 1)
            parsed[key] = value.strip('"')
        if parsed == labels:
            return float(value_part.strip())
    return 0.0


def test_metrics_endpoint_available():
    print("test_metrics_endpoint_available: build app", flush=True)
    app = create_app(
        database=None,
        pylet_client=None,
        router=None,
        autoscaler=None,
        config=None,
    )
    print("test_metrics_endpoint_available: enter TestClient", flush=True)
    with TestClient(app) as client:
        print("test_metrics_endpoint_available: GET /metrics", flush=True)
        response = client.get("/metrics")
        print(
            "test_metrics_endpoint_available: response",
            response.status_code,
            flush=True,
        )
    assert response.status_code == 200
    assert re.search(r"^# HELP ", response.text, flags=re.MULTILINE)


@pytest.mark.asyncio
async def test_router_metrics_request_increments(database, mock_http_client):
    from sllm.router import Router, RouterConfig

    deployment_id = "metrics-model:vllm"
    database.add_deployment_endpoint(deployment_id, "127.0.0.1:8000")

    metrics = get_metrics()
    router = Router(
        database=database,
        config=RouterConfig(),
        prometheus_metrics=metrics,
    )
    router._session = mock_http_client

    before = _metric_value(
        render_metrics()[0].decode("utf-8"),
        "sllm_router_requests_total",
        {"deployment_id": deployment_id, "backend": "vllm"},
    )
    await router.handle_request(
        {"model": "test"},
        "/v1/chat/completions",
        deployment_id=deployment_id,
    )
    after = _metric_value(
        render_metrics()[0].decode("utf-8"),
        "sllm_router_requests_total",
        {"deployment_id": deployment_id, "backend": "vllm"},
    )
    assert after >= before + 1


def test_router_instance_gauge_updates():
    deployment_id = "scale-model:vllm"
    metrics = get_metrics()

    metrics.set_instance_count(deployment_id, 2)
    value = _metric_value(
        render_metrics()[0].decode("utf-8"),
        "sllm_router_instances",
        {"deployment_id": deployment_id, "backend": "vllm"},
    )
    assert value == 2

    metrics.set_instance_count(deployment_id, 0)
    value = _metric_value(
        render_metrics()[0].decode("utf-8"),
        "sllm_router_instances",
        {"deployment_id": deployment_id, "backend": "vllm"},
    )
    assert value == 0


def test_prometheus_file_sd_writes_targets(tmp_path):
    sd_path = tmp_path / "sd.json"
    sd = PrometheusFileSD(str(sd_path))

    endpoints = {"facebook/opt-1.3b:vllm": ["10.0.0.12:8000", "10.0.0.11:8000"]}
    sd.write_targets(endpoints)

    assert sd_path.exists()
    payload = json.loads(sd_path.read_text())
    assert payload[0]["targets"] == ["10.0.0.11:8000", "10.0.0.12:8000"]
    assert payload[0]["labels"]["deployment_id"] == "facebook/opt-1.3b:vllm"
    assert payload[0]["labels"]["backend"] == "vllm"

    endpoints = {"facebook/opt-1.3b:vllm": ["10.0.0.12:8000"]}
    sd.write_targets(endpoints)

    payload = json.loads(sd_path.read_text())
    assert payload[0]["targets"] == ["10.0.0.12:8000"]
