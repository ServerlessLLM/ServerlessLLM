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
    VENV_TRANSFORMERS,
    VENV_VLLM,
    build_instance_command,
    build_sglang_command,
    build_transformers_command,
    build_vllm_command,
    get_command_builder,
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


class TestBuildSglangCommand:
    """Tests for SGLang command builder."""

    @pytest.fixture
    def sglang_model(self):
        """Create a mock SGLang model."""
        model = MagicMock(spec=Model)
        model.model_name = "meta-llama/Llama-2-7b"
        model.backend = "sglang"
        model.backend_config = {
            "tensor_parallel_size": 4,
            "max_model_len": 8192,
            "dtype": "bfloat16",
        }
        return model

    def test_basic_command(self, sglang_model):
        """Test basic SGLang command generation."""
        cmd, venv = build_sglang_command(sglang_model)

        assert "python -m sglang.launch_server" in cmd
        assert "--model-path /models/sglang/meta-llama/Llama-2-7b" in cmd
        assert "--served-model-name meta-llama/Llama-2-7b" in cmd
        assert "--port $PORT" in cmd
        assert "--host 0.0.0.0" in cmd
        assert venv == VENV_SGLANG

    def test_tensor_parallel(self, sglang_model):
        """Test tensor parallel in command."""
        cmd, _ = build_sglang_command(sglang_model)

        assert "--tp 4" in cmd

    def test_context_length(self, sglang_model):
        """Test context length mapping."""
        cmd, _ = build_sglang_command(sglang_model)

        # SGLang uses --context-length instead of --max-model-len
        assert "--context-length 8192" in cmd

    def test_dtype(self, sglang_model):
        """Test dtype in command."""
        cmd, _ = build_sglang_command(sglang_model)

        assert "--dtype bfloat16" in cmd

    def test_trust_remote_code(self, sglang_model):
        """Test trust remote code flag."""
        sglang_model.backend_config["trust_remote_code"] = True
        cmd, _ = build_sglang_command(sglang_model)

        assert "--trust-remote-code" in cmd


class TestBuildTransformersCommand:
    """Tests for Transformers command builder."""

    @pytest.fixture
    def transformers_model(self):
        """Create a mock Transformers model."""
        model = MagicMock(spec=Model)
        model.model_name = "gpt2"
        model.backend = "transformers"
        model.backend_config = {
            "dtype": "float32",
            "device_map": "cuda:0",
        }
        return model

    def test_basic_command(self, transformers_model):
        """Test basic Transformers command generation."""
        cmd, venv = build_transformers_command(transformers_model)

        assert "python -m sllm.backends.transformers_server" in cmd
        assert "--model-path /models/transformers/gpt2" in cmd
        assert "--served-model-name gpt2" in cmd
        assert "--port $PORT" in cmd
        assert "--host 0.0.0.0" in cmd
        assert venv == VENV_TRANSFORMERS

    def test_dtype(self, transformers_model):
        """Test dtype in command."""
        cmd, _ = build_transformers_command(transformers_model)

        assert "--dtype float32" in cmd

    def test_device_map(self, transformers_model):
        """Test device map in command."""
        cmd, _ = build_transformers_command(transformers_model)

        assert "--device-map cuda:0" in cmd

    def test_default_values(self):
        """Test default values when not specified."""
        model = MagicMock(spec=Model)
        model.model_name = "test"
        model.backend = "transformers"
        model.backend_config = {}

        cmd, _ = build_transformers_command(model)

        assert "--dtype float16" in cmd  # Default
        assert "--device-map auto" in cmd  # Default


class TestGetCommandBuilder:
    """Tests for get_command_builder function."""

    def test_get_vllm_builder(self):
        """Test getting vLLM command builder."""
        builder = get_command_builder("vllm")
        assert builder is build_vllm_command

    def test_get_sglang_builder(self):
        """Test getting SGLang command builder."""
        builder = get_command_builder("sglang")
        assert builder is build_sglang_command

    def test_get_transformers_builder(self):
        """Test getting Transformers command builder."""
        builder = get_command_builder("transformers")
        assert builder is build_transformers_command

    def test_unknown_backend(self):
        """Test getting builder for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_command_builder("unknown")


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

    def test_sglang_backend(self):
        """Test build_instance_command with SGLang backend."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "sglang"
        model.backend_config = {}

        cmd, venv = build_instance_command(model)

        assert "sglang.launch_server" in cmd
        assert venv == VENV_SGLANG

    def test_transformers_backend(self):
        """Test build_instance_command with Transformers backend."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "transformers"
        model.backend_config = {}

        cmd, venv = build_instance_command(model)

        assert "sllm.backends.transformers_server" in cmd
        assert venv == VENV_TRANSFORMERS

    def test_unknown_backend(self):
        """Test build_instance_command with unknown backend."""
        model = MagicMock(spec=Model)
        model.model_name = "test-model"
        model.backend = "unknown"
        model.backend_config = {}

        with pytest.raises(ValueError, match="Unknown backend"):
            build_instance_command(model)
