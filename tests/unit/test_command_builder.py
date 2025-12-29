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
    VENV_VLLM,
    build_instance_command,
    build_vllm_command,
)
from sllm.database import Model


class TestBuildVllmCommand:
    """Tests for vLLM command builder."""

    @pytest.fixture
    def vllm_model(self):
        """Create a mock vLLM model."""
        model = MagicMock(spec=Model)
        model.model_name = "facebook/opt-125m"
        model.backend = "vllm"
        model.backend_config = {
            "tensor_parallel_size": 2,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.9,
            "dtype": "float16",
        }
        return model

    def test_basic_command(self, vllm_model):
        """Test basic vLLM command generation."""
        cmd, venv = build_vllm_command(vllm_model)

        assert "vllm serve" in cmd
        assert "/models/vllm/facebook/opt-125m" in cmd
        assert "--load-format serverless_llm" in cmd
        assert "--served-model-name facebook/opt-125m" in cmd
        assert "--port $PORT" in cmd
        assert "--host 0.0.0.0" in cmd
        assert venv == VENV_VLLM

    def test_tensor_parallel_size(self, vllm_model):
        """Test tensor parallel size in command."""
        cmd, _ = build_vllm_command(vllm_model)

        assert "--tensor-parallel-size 2" in cmd

    def test_max_model_len(self, vllm_model):
        """Test max model length in command."""
        cmd, _ = build_vllm_command(vllm_model)

        assert "--max-model-len 4096" in cmd

    def test_gpu_memory_utilization(self, vllm_model):
        """Test GPU memory utilization in command."""
        cmd, _ = build_vllm_command(vllm_model)

        assert "--gpu-memory-utilization 0.9" in cmd

    def test_dtype(self, vllm_model):
        """Test dtype in command."""
        cmd, _ = build_vllm_command(vllm_model)

        assert "--dtype float16" in cmd

    def test_trust_remote_code(self, vllm_model):
        """Test trust remote code flag."""
        vllm_model.backend_config["trust_remote_code"] = True
        cmd, _ = build_vllm_command(vllm_model)

        assert "--trust-remote-code" in cmd

    def test_extra_args_list(self, vllm_model):
        """Test extra args as list."""
        vllm_model.backend_config["extra_args"] = [
            "--enable-chunked-prefill",
            "--max-num-seqs 128",
        ]
        cmd, _ = build_vllm_command(vllm_model)

        assert "--enable-chunked-prefill" in cmd
        assert "--max-num-seqs 128" in cmd

    def test_extra_args_string(self, vllm_model):
        """Test extra args as string."""
        vllm_model.backend_config["extra_args"] = "--enable-chunked-prefill"
        cmd, _ = build_vllm_command(vllm_model)

        assert "--enable-chunked-prefill" in cmd

    def test_custom_storage_path(self, vllm_model):
        """Test custom storage path."""
        cmd, _ = build_vllm_command(vllm_model, storage_path="/custom/path")

        assert "/custom/path/vllm/facebook/opt-125m" in cmd

    def test_minimal_config(self):
        """Test with minimal backend config."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "vllm"
        model.backend_config = {}

        cmd, _ = build_vllm_command(model)

        assert "vllm serve" in cmd
        assert "--tensor-parallel-size 1" in cmd  # Default
        assert "--max-model-len" not in cmd  # Not set
        assert "--gpu-memory-utilization" not in cmd  # Not set

    def test_none_backend_config(self):
        """Test with None backend config."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "vllm"
        model.backend_config = None

        cmd, _ = build_vllm_command(model)

        assert "vllm serve" in cmd


class TestBuildInstanceCommand:
    """Tests for build_instance_command convenience function."""

    def test_vllm_backend(self):
        """Test build_instance_command with vLLM backend."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "vllm"
        model.backend_config = {}

        cmd, venv = build_instance_command(model)

        assert "vllm serve" in cmd
        assert venv == VENV_VLLM

    def test_unknown_backend(self):
        """Test build_instance_command with unknown backend."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "unknown"
        model.backend_config = {}

        with pytest.raises(ValueError, match="Unknown backend"):
            build_instance_command(model)
