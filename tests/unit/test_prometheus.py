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

from sllm.prometheus import PrometheusFileSD


def test_prometheus_file_sd_writes_targets(tmp_path):
    sd_path = tmp_path / "sd.json"
    sd = PrometheusFileSD(str(sd_path))

    endpoints = {"facebook/opt-1.3b:vllm": ["10.0.0.12:8000", "10.0.0.11:8000"]}
    sd.write_targets(endpoints)

    payload = json.loads(sd_path.read_text())
    assert payload[0]["targets"] == ["10.0.0.11:8000", "10.0.0.12:8000"]
    assert payload[0]["labels"]["deployment_id"] == "facebook/opt-1.3b:vllm"
    assert payload[0]["labels"]["backend"] == "vllm"
