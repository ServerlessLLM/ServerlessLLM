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

from sllm.database import Deployment

VENV_VLLM = "/opt/venvs/vllm"
VENV_SGLANG = "/opt/venvs/sglang"
VENV_SLLM_STORE = "/opt/venvs/sllm-store"

# Backend install instructions
BACKEND_INSTALL_INSTRUCTIONS = {
    "vllm": "pip install vllm",
    "sglang": "pip install 'sglang[all]'",
}


def check_backend_available(backend: str) -> None:
    """
    Check if a backend is available (importable).

    Raises ImportError with helpful message if not installed.
    Note: This checks the current Python environment. On distributed setups,
    backends run in separate venvs on worker nodes.

    Args:
        backend: Backend name ("vllm" or "sglang")

    Raises:
        ImportError: If backend is not installed
        ValueError: If backend is unknown
    """
    if backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            raise ImportError(
                "vLLM is required for the vllm backend but is not installed. "
                "Install it with: pip install vllm"
            )
    elif backend == "sglang":
        try:
            import sglang  # noqa: F401
        except ImportError:
            raise ImportError(
                f"SGLang is required for the sglang backend but is not installed. "
                f"Install it with: {BACKEND_INSTALL_INSTRUCTIONS['sglang']}"
            )
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: vllm, sglang"
        )


def build_vllm_command(
    deployment: Deployment, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build vLLM serve command for Pylet submission."""
    config = deployment.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)
    max_model_len = config.get("max_model_len")
    gpu_memory_utilization = config.get("gpu_memory_utilization")
    dtype = config.get("dtype")
    trust_remote_code = config.get("trust_remote_code", False)

    cmd_parts = [
        "vllm serve",
        deployment.model_name,
        f"--served-model-name {deployment.model_name}",
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


def build_sglang_command(
    deployment: Deployment, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build SGLang launch_server command for Pylet submission."""
    config = deployment.backend_config or {}
    tp = config.get("tensor_parallel_size", 1)

    cmd_parts = [
        "python -m sglang.launch_server",
        f"--model-path {deployment.model_name}",
        f"--served-model-name {deployment.model_name}",
        "--port $PORT",
        "--host 0.0.0.0",
        f"--tp {tp}",
    ]

    if config.get("mem_fraction_static"):
        cmd_parts.append(
            f"--mem-fraction-static {config['mem_fraction_static']}"
        )

    if config.get("dtype"):
        cmd_parts.append(f"--dtype {config['dtype']}")

    if config.get("trust_remote_code"):
        cmd_parts.append("--trust-remote-code")

    extra_args = config.get("extra_args", [])
    if extra_args:
        if isinstance(extra_args, list):
            cmd_parts.extend(extra_args)
        elif isinstance(extra_args, str):
            cmd_parts.append(extra_args)

    return " ".join(cmd_parts), VENV_SGLANG


BUILDERS = {
    "vllm": build_vllm_command,
    "sglang": build_sglang_command,
}


def build_instance_command(
    deployment: Deployment, storage_path: str = "/models"
) -> Tuple[str, str]:
    """Build command for a deployment instance."""
    builder = BUILDERS.get(deployment.backend)
    if not builder:
        raise ValueError(f"Unknown backend: {deployment.backend}")
    return builder(deployment, storage_path)
