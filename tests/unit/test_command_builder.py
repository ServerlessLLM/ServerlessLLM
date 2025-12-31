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
"""Tests for command builders."""

from unittest.mock import MagicMock

import pytest

from sllm.command_builder import (
    VENV_SGLANG,
    VENV_VLLM,
    build_instance_command,
    build_sglang_command,
    build_vllm_command,
)
from sllm.database import Deployment


class TestBuildVllmCommand:
    """Tests for vLLM command builder."""

    @pytest.fixture
    def vllm_deployment(self):
        """Create a mock vLLM deployment."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "facebook/opt-125m"
        deployment.backend = "vllm"
        deployment.backend_config = {
            "tensor_parallel_size": 2,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "dtype": "float16",
        }
        return deployment

    def test_basic_command(self, vllm_deployment):
        """Test basic vLLM command generation."""
        cmd, venv = build_vllm_command(vllm_deployment)

        assert "vllm serve" in cmd
        assert "facebook/opt-125m" in cmd
        assert "--served-model-name facebook/opt-125m" in cmd
        assert "--port $PORT" in cmd
        assert "--host 0.0.0.0" in cmd
        assert venv == VENV_VLLM

    def test_tensor_parallel_size(self, vllm_deployment):
        """Test tensor parallel size in command."""
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--tensor-parallel-size 2" in cmd

    def test_max_model_len(self, vllm_deployment):
        """Test max model length in command."""
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--max-model-len 4096" in cmd

    def test_gpu_memory_utilization(self, vllm_deployment):
        """Test GPU memory utilization in command."""
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--gpu-memory-utilization 0.9" in cmd

    def test_dtype(self, vllm_deployment):
        """Test dtype in command."""
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--dtype float16" in cmd

    def test_trust_remote_code(self, vllm_deployment):
        """Test trust remote code flag."""
        vllm_deployment.backend_config["trust_remote_code"] = True
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--trust-remote-code" in cmd

    def test_extra_args_list(self, vllm_deployment):
        """Test extra args as list."""
        vllm_deployment.backend_config["extra_args"] = [
            "--enable-chunked-prefill",
            "--max-num-seqs 128",
        ]
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--enable-chunked-prefill" in cmd
        assert "--max-num-seqs 128" in cmd

    def test_extra_args_string(self, vllm_deployment):
        """Test extra args as string."""
        vllm_deployment.backend_config["extra_args"] = (
            "--enable-chunked-prefill"
        )
        cmd, _ = build_vllm_command(vllm_deployment)

        assert "--enable-chunked-prefill" in cmd

    def test_minimal_config(self):
        """Test with minimal backend config."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "vllm"
        deployment.backend_config = {}

        cmd, _ = build_vllm_command(deployment)

        assert "vllm serve" in cmd
        assert "--tensor-parallel-size 1" in cmd  # Default
        assert "--max-model-len" not in cmd  # Not set
        assert "--gpu-memory-utilization" not in cmd  # Not set

    def test_none_backend_config(self):
        """Test with None backend config."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "vllm"
        deployment.backend_config = None

        cmd, _ = build_vllm_command(deployment)

        assert "vllm serve" in cmd


class TestBuildSglangCommand:
    """Tests for SGLang command builder."""

    @pytest.fixture
    def sglang_deployment(self):
        """Create a mock SGLang deployment."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "meta-llama/Llama-3.1-8B"
        deployment.backend = "sglang"
        deployment.backend_config = {
            "tensor_parallel_size": 2,
            "mem_fraction_static": 0.8,
            "dtype": "float16",
        }
        return deployment

    def test_basic_command(self, sglang_deployment):
        """Test basic SGLang command generation."""
        cmd, venv = build_sglang_command(sglang_deployment)

        assert "python -m sglang.launch_server" in cmd
        assert "--model-path meta-llama/Llama-3.1-8B" in cmd
        assert "--served-model-name meta-llama/Llama-3.1-8B" in cmd
        assert "--port $PORT" in cmd
        assert "--host 0.0.0.0" in cmd
        assert venv == VENV_SGLANG

    def test_tensor_parallel_size(self, sglang_deployment):
        """Test tensor parallel size in command (uses --tp shorthand)."""
        cmd, _ = build_sglang_command(sglang_deployment)

        assert "--tp 2" in cmd

    def test_mem_fraction_static(self, sglang_deployment):
        """Test memory fraction static in command."""
        cmd, _ = build_sglang_command(sglang_deployment)

        assert "--mem-fraction-static 0.8" in cmd

    def test_dtype(self, sglang_deployment):
        """Test dtype in command."""
        cmd, _ = build_sglang_command(sglang_deployment)

        assert "--dtype float16" in cmd

    def test_trust_remote_code(self, sglang_deployment):
        """Test trust remote code flag."""
        sglang_deployment.backend_config["trust_remote_code"] = True
        cmd, _ = build_sglang_command(sglang_deployment)

        assert "--trust-remote-code" in cmd

    def test_extra_args_list(self, sglang_deployment):
        """Test extra args as list."""
        sglang_deployment.backend_config["extra_args"] = [
            "--chunked-prefill-size",
            "8192",
        ]
        cmd, _ = build_sglang_command(sglang_deployment)

        assert "--chunked-prefill-size" in cmd
        assert "8192" in cmd

    def test_extra_args_string(self, sglang_deployment):
        """Test extra args as string."""
        sglang_deployment.backend_config["extra_args"] = "--disable-radix-cache"
        cmd, _ = build_sglang_command(sglang_deployment)

        assert "--disable-radix-cache" in cmd

    def test_minimal_config(self):
        """Test with minimal backend config."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "sglang"
        deployment.backend_config = {}

        cmd, _ = build_sglang_command(deployment)

        assert "python -m sglang.launch_server" in cmd
        assert "--tp 1" in cmd  # Default
        assert "--mem-fraction-static" not in cmd  # Not set

    def test_none_backend_config(self):
        """Test with None backend config."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "sglang"
        deployment.backend_config = None

        cmd, _ = build_sglang_command(deployment)

        assert "python -m sglang.launch_server" in cmd


class TestBuildInstanceCommand:
    """Tests for build_instance_command dispatcher."""

    def test_vllm_backend(self):
        """Test build_instance_command with vLLM backend."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "vllm"
        deployment.backend_config = {}

        cmd, venv = build_instance_command(deployment)

        assert "vllm serve" in cmd
        assert venv == VENV_VLLM

    def test_sglang_backend(self):
        """Test build_instance_command with SGLang backend."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "sglang"
        deployment.backend_config = {}

        cmd, venv = build_instance_command(deployment)

        assert "sglang.launch_server" in cmd
        assert venv == VENV_SGLANG

    def test_unknown_backend(self):
        """Test build_instance_command with unknown backend."""
        deployment = MagicMock(spec=Deployment)
        deployment.model_name = "test-model"
        deployment.backend = "unknown"
        deployment.backend_config = {}

        with pytest.raises(ValueError, match="Unknown backend"):
            build_instance_command(deployment)
