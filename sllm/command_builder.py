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
"""Command builders for ServerlessLLM inference backends."""

from typing import Tuple

from sllm.database import Model

VENV_VLLM = "/opt/venvs/vllm"
VENV_SLLM_STORE = "/opt/venvs/sllm-store"


def build_vllm_command(
    model: Model, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build vLLM serve command for Pylet submission."""
    config = model.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)
    max_model_len = config.get("max_model_len")
    gpu_memory_utilization = config.get("gpu_memory_utilization")
    dtype = config.get("dtype")
    trust_remote_code = config.get("trust_remote_code", False)

    model_path = f"{storage_path}/vllm/{model.model_name}"

    cmd_parts = [
        "vllm serve",
        model_path,
        "--load-format serverless_llm",
        f"--served-model-name {model.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--tensor-parallel-size {tp}",
    ]

    if max_model_len:
        cmd_parts.append(f"--max-model-len {max_model_len}")

    if gpu_memory_utilization:
        cmd_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")

    if dtype:
        cmd_parts.append(f"--dtype {dtype}")

    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    extra_args = config.get("extra_args", [])
    if extra_args:
        if isinstance(extra_args, list):
            cmd_parts.extend(extra_args)
        elif isinstance(extra_args, str):
            cmd_parts.append(extra_args)

    return " ".join(cmd_parts), VENV_VLLM


def build_instance_command(
    model: Model, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build command for a model instance."""
    if model.backend == "vllm":
        return build_vllm_command(model, storage_path)
    raise ValueError(f"Unknown backend: {model.backend}")
