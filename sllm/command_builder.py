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
Command builders for ServerlessLLM v1-beta.

Builds shell commands for starting vLLM and SGLang inference backends
via Pylet submission. Each builder returns (command, venv_path) tuple
for use with Pylet's --venv parameter.
"""

import os
from typing import Tuple

from sllm.database import Model

# Venv paths for different backends (used with Pylet --venv parameter)
VENV_VLLM = os.environ.get("SLLM_VENV_VLLM", "/opt/venvs/vllm")
VENV_SGLANG = os.environ.get("SLLM_VENV_SGLANG", "/opt/venvs/sglang")
VENV_TRANSFORMERS = os.environ.get("SLLM_VENV_TRANSFORMERS", "/opt/venvs/vllm")


def build_vllm_command(
    model: Model, storage_path: str = "/models"
) -> Tuple[str, str]:
    """
    Build vLLM serve command for Pylet submission.

    Args:
        model: Model configuration from database
        storage_path: Path to model storage

    Returns:
        Tuple of (shell command string, venv path)
    """
    config = model.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)
    max_model_len = config.get("max_model_len")
    gpu_memory_utilization = config.get("gpu_memory_utilization")
    dtype = config.get("dtype")
    trust_remote_code = config.get("trust_remote_code", False)

    # Model path: /models/vllm/{model_name}
    model_path = f"{storage_path}/vllm/{model.model_name}"

    # Base command
    cmd_parts = [
        "vllm serve",
        model_path,
        "--load-format serverless_llm",
        f"--served-model-name {model.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--tensor-parallel-size {tp}",
    ]

    # Optional parameters
    if max_model_len:
        cmd_parts.append(f"--max-model-len {max_model_len}")

    if gpu_memory_utilization:
        cmd_parts.append(f"--gpu-memory-utilization {gpu_memory_utilization}")

    if dtype:
        cmd_parts.append(f"--dtype {dtype}")

    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    # Additional arbitrary arguments
    extra_args = config.get("extra_args", [])
    if extra_args:
        if isinstance(extra_args, list):
            cmd_parts.extend(extra_args)
        elif isinstance(extra_args, str):
            cmd_parts.append(extra_args)

    return " ".join(cmd_parts), VENV_VLLM


def build_sglang_command(
    model: Model, storage_path: str = "/models"
) -> Tuple[str, str]:
    """
    Build SGLang serve command for Pylet submission.

    Args:
        model: Model configuration from database
        storage_path: Path to model storage

    Returns:
        Tuple of (shell command string, venv path)
    """
    config = model.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)
    max_model_len = config.get("max_model_len")
    dtype = config.get("dtype")
    trust_remote_code = config.get("trust_remote_code", False)

    # Model path: /models/sglang/{model_name}
    model_path = f"{storage_path}/sglang/{model.model_name}"

    # Base command
    cmd_parts = [
        "python -m sglang.launch_server",
        f"--model-path {model_path}",
        f"--served-model-name {model.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--tp {tp}",
    ]

    # Optional parameters
    if max_model_len:
        cmd_parts.append(f"--context-length {max_model_len}")

    if dtype:
        cmd_parts.append(f"--dtype {dtype}")

    if trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    # Additional arbitrary arguments
    extra_args = config.get("extra_args", [])
    if extra_args:
        if isinstance(extra_args, list):
            cmd_parts.extend(extra_args)
        elif isinstance(extra_args, str):
            cmd_parts.append(extra_args)

    return " ".join(cmd_parts), VENV_SGLANG


def build_transformers_command(
    model: Model, storage_path: str = "/models"
) -> Tuple[str, str]:
    """
    Build Transformers backend command for Pylet submission.

    Args:
        model: Model configuration from database
        storage_path: Path to model storage

    Returns:
        Tuple of (shell command string, venv path)
    """
    config = model.backend_config or {}
    dtype = config.get("dtype", "float16")
    device_map = config.get("device_map", "auto")

    # Model path
    model_path = f"{storage_path}/transformers/{model.model_name}"

    # Use sllm's transformers backend server
    cmd_parts = [
        "python -m sllm.backends.transformers_server",
        f"--model-path {model_path}",
        f"--served-model-name {model.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--dtype {dtype}",
        f"--device-map {device_map}",
    ]

    return " ".join(cmd_parts), VENV_TRANSFORMERS


def get_command_builder(backend: str):
    """
    Get the appropriate command builder for a backend.

    Args:
        backend: Backend type ("vllm", "sglang", "transformers")

    Returns:
        Command builder function that returns (command, venv_path) tuple
    """
    builders = {
        "vllm": build_vllm_command,
        "sglang": build_sglang_command,
        "transformers": build_transformers_command,
    }

    builder = builders.get(backend)
    if not builder:
        raise ValueError(f"Unknown backend: {backend}")

    return builder


def build_instance_command(
    model: Model, storage_path: str = "/models"
) -> Tuple[str, str]:
    """
    Build command for a model instance.

    Convenience function that selects the right builder based on backend.

    Args:
        model: Model configuration from database
        storage_path: Path to model storage

    Returns:
        Tuple of (command, venv_path)
    """
    builder = get_command_builder(model.backend)
    return builder(model, storage_path)
