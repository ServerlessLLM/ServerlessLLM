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
import os
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from sllm.backends.backend_utils import BackendStatus
from sllm.backends.transformers_backend import (
    TransformersBackend,
)


@pytest.fixture
def model_name():
    return "facebook/opt-125m"


@pytest.fixture
def encoder_model_name():
    return "BAAI/bge-small-en-v1.5"


@pytest.fixture
def backend_config():
    return {
        "pretrained_model_name_or_path": "facebook/opt-125m",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
    }


@pytest.fixture
def encoder_config():
    return {
        "pretrained_model_name_or_path": "BAAI/bge-small-en-v1.5",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModel",
    }


@pytest.fixture
def transformers_backend(model_name, backend_config):
    yield TransformersBackend(model_name, backend_config)


@pytest.fixture
def encoder_backend(encoder_model_name, encoder_config):
    yield TransformersBackend(encoder_model_name, encoder_config)


def test_init(transformers_backend, model_name, backend_config):
    assert transformers_backend.model_name == model_name
    assert transformers_backend.backend_config == backend_config
    assert transformers_backend.status == BackendStatus.UNINITIALIZED


def test_init_encoder(encoder_backend, encoder_model_name, encoder_config):
    assert encoder_backend.model_name == encoder_model_name
    assert encoder_backend.backend_config == encoder_config
    assert encoder_backend.status == BackendStatus.UNINITIALIZED


def test_init_backend(transformers_backend, backend_config):
    with patch(
        "sllm.backends.transformers_backend.load_model"
    ) as mock_load_model:
        transformers_backend.init_backend()
        mock_load_model.assert_called_once()
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(
            "transformers",
            backend_config["pretrained_model_name_or_path"],
        )
        device_map = backend_config.get("device_map", "auto")
        torch_dtype = backend_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        hf_model_class = backend_config.get("hf_model_class", None)
        quantization_config = backend_config.get("quantization_config", None)
        mock_load_model.assert_called_once_with(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
            hf_model_class=hf_model_class,
            quantization_config=quantization_config,
        )


def test_init_encoder_backend(encoder_backend, encoder_config):
    with patch(
        "sllm.backends.transformers_backend.load_model"
    ) as mock_load_model:
        encoder_backend.init_backend()
        mock_load_model.assert_called_once()
        storage_path = os.getenv("STORAGE_PATH", "./models")
        model_path = os.path.join(
            "transformers",
            encoder_config["pretrained_model_name_or_path"],
        )
        device_map = encoder_config.get("device_map", "auto")
        torch_dtype = encoder_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        hf_model_class = encoder_config.get("hf_model_class", None)
        quantization_config = encoder_config.get("quantization_config", None)
        mock_load_model.assert_called_once_with(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
            hf_model_class=hf_model_class,
            quantization_config=quantization_config,
        )


@pytest.fixture
def model():
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to("cpu")
    yield model


@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    yield tokenizer("test_prompt", return_tensors="pt")


@pytest.fixture
def encoder():
    encoder = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to("cpu")
    yield encoder


@pytest.fixture
def encoder_tokenizer():
    encoder_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    yield encoder_tokenizer(
        ["test_prompt"],
        max_length=4096,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )


def test_generate(transformers_backend, model, tokenizer):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        transformers_backend.init_backend()
        input = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
        }
        result = transformers_backend.generate(input)
        assert "error" not in result
        assert "choices" in result and len(result["choices"]) == 1


def test_get_current_tokens(transformers_backend, model, tokenizer):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        transformers_backend.init_backend()
        input = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 128,
        }
        import threading

        # Create a thread to call the generate method
        thread = threading.Thread(
            target=transformers_backend.generate, args=(input,)
        )
        thread.start()
        # Sleep for 1 second to allow the thread to start
        import time

        time.sleep(1)
        # Get the current tokens
        current_tokens = transformers_backend.get_current_tokens()
        assert current_tokens
        thread.join()


def test_resume_kv_cache(transformers_backend, model, tokenizer):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        transformers_backend.init_backend()
        inputs = transformers_backend._tokenize("")
        intermediate_tokens = inputs["input_ids"].tolist()
        try:
            transformers_backend.resume_kv_cache(intermediate_tokens)
        except Exception as e:
            assert False, f"Failed to resume kv cache: {e}"
        assert transformers_backend.past_key_values
        assert (
            len(transformers_backend.past_key_values)
            == model.config.num_hidden_layers
        )
        assert len(transformers_backend.past_key_values[0]) == 2
        assert transformers_backend.past_key_values[0][0].shape == (
            len(intermediate_tokens),
            model.config.num_attention_heads,
            len(intermediate_tokens[0]),
            model.config.hidden_size // model.config.num_attention_heads,
        )


def test_resume_generate(transformers_backend, model, tokenizer):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        transformers_backend.init_backend()
        input = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 128,
        }
        intermediate_inputs = transformers_backend._tokenize("")
        intermediate_tokens = intermediate_inputs["input_ids"].tolist()
        try:
            transformers_backend.resume_kv_cache(intermediate_tokens)
        except Exception as e:
            assert False, f"Failed to resume kv cache: {e}"

        result = transformers_backend.resume_generate(
            input, intermediate_tokens
        )
        assert result
        assert "error" not in result


def test_shutdown(transformers_backend, model, tokenizer):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        transformers_backend.init_backend()
        input = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 128,
        }
        import threading

        # Create a thread to call the generate method
        thread = threading.Thread(
            target=transformers_backend.generate, args=(input,)
        )
        thread.start()
        # Sleep for 1 second to allow the thread to start
        import time

        time.sleep(1)
        # Shutdown the backend
        transformers_backend.shutdown()
        thread.join()


def test_encode(encoder_backend, encoder, encoder_tokenizer):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=encoder,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._encoder_tokenize",
        return_value=encoder_tokenizer,
    ):
        encoder_backend.init_backend()
        input = {
            "model": "BAAI/bge-small-en-v1.5",
            "task_instruct": "Given a question, retrieve passages that answer the question",
            "input": ["Hi, How are you?"],
        }
        result = encoder_backend.encode(input)
        assert "error" not in result
        assert "data" in result and len(result["data"]) == 1


def test_generate_without_init(transformers_backend):
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=model,
    ):
        request_data = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you? I am fine, thank you!",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
        }
        response = transformers_backend.generate(request_data)
        assert "error" in response


# LoRA-related fixtures and tests
@pytest.fixture
def base_model_name():
    return "facebook/opt-125m"


@pytest.fixture
def lora_model_name():
    return "peft-internal-testing/opt-125m-dummy-lora"


@pytest.fixture
def lora_model_name_2():
    return "monsterapi/opt125M_alpaca"


@pytest.fixture
def lora_backend_config():
    return {
        "pretrained_model_name_or_path": "facebook/opt-125m",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
        "device_map": "cpu",  # Use CPU for testing to avoid GPU memory issues
    }


@pytest.fixture
def lora_backend(base_model_name, lora_backend_config):
    backend = TransformersBackend(base_model_name, lora_backend_config)
    yield backend


@pytest.fixture
def mock_peft_model():
    """Create a mock PEFT model with the necessary attributes."""
    from unittest.mock import MagicMock

    model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to("cpu")

    # Add PEFT-like attributes
    model.peft_config = {}
    model.add_adapter = MagicMock()
    model.set_adapter = MagicMock()

    return model


def test_load_lora_adapter_success(lora_backend, mock_peft_model):
    """Test successful loading of a LoRA adapter."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.load_lora",
        return_value=mock_peft_model,
    ) as mock_load_lora:
        lora_backend.init_backend()

        # Simulate successful LoRA loading by updating peft_config
        def mock_load_lora_side_effect(*args, **kwargs):
            mock_peft_model.peft_config["test_lora"] = {
                "adapter_name": "test_lora"
            }
            return mock_peft_model

        mock_load_lora.side_effect = mock_load_lora_side_effect

        # Load the LoRA adapter
        result = lora_backend.load_lora_adapter("test_lora", "dummy_path")

        # Verify the adapter is loaded
        assert hasattr(lora_backend.model, "peft_config")
        assert "test_lora" in lora_backend.model.peft_config
        mock_load_lora.assert_called_once()


def test_load_lora_adapter_repeated(lora_backend, mock_peft_model):
    """Test that loading the same LoRA adapter multiple times doesn't cause issues."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.load_lora",
        return_value=mock_peft_model,
    ) as mock_load_lora:
        lora_backend.init_backend()

        # Simulate LoRA already loaded
        mock_peft_model.peft_config["test_lora"] = {"adapter_name": "test_lora"}

        # Load the LoRA adapter first time (should be skipped)
        result = lora_backend.load_lora_adapter("test_lora", "dummy_path")

        # Load the same adapter again - should not call load_lora
        result = lora_backend.load_lora_adapter("test_lora", "dummy_path")

        # Verify load_lora was not called since adapter already exists
        mock_load_lora.assert_not_called()

        # Verify the adapter is still loaded
        assert hasattr(lora_backend.model, "peft_config")
        assert "test_lora" in lora_backend.model.peft_config


def test_load_multiple_lora_adapters(lora_backend, mock_peft_model):
    """Test loading multiple LoRA adapters."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.load_lora",
        return_value=mock_peft_model,
    ) as mock_load_lora:
        lora_backend.init_backend()

        # Simulate successful LoRA loading by updating peft_config
        def mock_load_lora_side_effect(*args, **kwargs):
            adapter_name = args[1]  # Second argument is adapter_name
            mock_peft_model.peft_config[adapter_name] = {
                "adapter_name": adapter_name
            }
            return mock_peft_model

        mock_load_lora.side_effect = mock_load_lora_side_effect

        # Load first LoRA adapter
        lora_backend.load_lora_adapter("test_lora_1", "dummy_path_1")

        # Load second LoRA adapter
        lora_backend.load_lora_adapter("test_lora_2", "dummy_path_2")

        # Verify both adapters are loaded
        assert hasattr(lora_backend.model, "peft_config")
        assert "test_lora_1" in lora_backend.model.peft_config
        assert "test_lora_2" in lora_backend.model.peft_config
        assert mock_load_lora.call_count == 2


def test_generate_with_lora_adapter(lora_backend, mock_peft_model):
    """Test generation with a loaded LoRA adapter."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.load_lora",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize"
    ) as mock_tokenize:
        lora_backend.init_backend()

        # Setup tokenizer mock
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        mock_inputs = tokenizer("test prompt", return_tensors="pt")
        mock_tokenize.return_value = mock_inputs

        # Simulate LoRA adapter loaded
        mock_peft_model.peft_config["test_lora"] = {"adapter_name": "test_lora"}

        # Mock model generation
        with patch.object(mock_peft_model, "generate") as mock_generate:
            # Create mock output tokens
            mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_generate.return_value = mock_output

            # Generate with LoRA adapter
            input_data = {
                "model": "facebook/opt-125m",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 10,
                "lora_adapter_name": "test_lora",
            }

            result = lora_backend.generate(input_data)

            # Verify successful generation
            assert "error" not in result
            assert "choices" in result and len(result["choices"]) == 1
            assert "message" in result["choices"][0]
            assert "content" in result["choices"][0]["message"]

            # Verify generate was called with adapter_names
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert "adapter_names" in call_kwargs
            assert call_kwargs["adapter_names"] == ["test_lora"]


def test_generate_with_unloaded_lora_adapter(lora_backend, mock_peft_model):
    """Test generation with an unloaded LoRA adapter should fail."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ):
        lora_backend.init_backend()

        input_data = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
            "lora_adapter_name": "nonexistent_lora",
        }

        result = lora_backend.generate(input_data)

        # Verify error is returned
        assert "error" in result
        assert "LoRA adapter nonexistent_lora not found" in result["error"]


def test_generate_base_model_without_lora(lora_backend, mock_peft_model):
    """Test generation with base model (no LoRA adapter specified)."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize"
    ) as mock_tokenize:
        lora_backend.init_backend()

        # Setup tokenizer mock
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        mock_inputs = tokenizer("test prompt", return_tensors="pt")
        mock_tokenize.return_value = mock_inputs

        # Mock model generation
        with patch.object(mock_peft_model, "generate") as mock_generate:
            # Create mock output tokens
            mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_generate.return_value = mock_output

            input_data = {
                "model": "facebook/opt-125m",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 10,
            }

            result = lora_backend.generate(input_data)

            # Verify successful generation
            assert "error" not in result
            assert "choices" in result and len(result["choices"]) == 1
            assert "message" in result["choices"][0]
            assert "content" in result["choices"][0]["message"]

            # Verify generate was called without adapter_names
            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs.get("adapter_names") is None


def test_generate_with_lora_on_base_model_error(lora_backend, mock_peft_model):
    """Test that requesting LoRA generation on a base model without LoRA loaded fails."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ):
        lora_backend.init_backend()

        # Don't load any LoRA adapter (peft_config is empty)

        input_data = {
            "model": "facebook/opt-125m",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello, how are you?",
                }
            ],
            "temperature": 0.7,
            "max_tokens": 10,
            "lora_adapter_name": "test_lora",
        }

        result = lora_backend.generate(input_data)

        # Verify error is returned
        assert "error" in result
        assert "LoRA adapter test_lora not found" in result["error"]


def test_load_lora_adapter_uninitialized_backend():
    """Test that loading LoRA adapter on uninitialized backend fails."""
    backend_config = {
        "pretrained_model_name_or_path": "facebook/opt-125m",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
    }
    backend = TransformersBackend("facebook/opt-125m", backend_config)

    # Try to load LoRA without initializing backend
    result = backend.load_lora_adapter("test_lora", "dummy_path")

    # Verify error is returned
    assert result is not None
    assert "error" in result
    assert "Model not initialized" in result["error"]


def test_generate_with_different_lora_adapters(lora_backend, mock_peft_model):
    """Test generation with different LoRA adapters."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.load_lora",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize"
    ) as mock_tokenize:
        lora_backend.init_backend()

        # Setup tokenizer mock
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        mock_inputs = tokenizer("test prompt", return_tensors="pt")
        mock_tokenize.return_value = mock_inputs

        # Simulate both LoRA adapters loaded
        mock_peft_model.peft_config["test_lora_1"] = {
            "adapter_name": "test_lora_1"
        }
        mock_peft_model.peft_config["test_lora_2"] = {
            "adapter_name": "test_lora_2"
        }

        # Mock model generation
        with patch.object(mock_peft_model, "generate") as mock_generate:
            # Create mock output tokens
            mock_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            mock_generate.return_value = mock_output

            input_data_base = {
                "model": "facebook/opt-125m",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ],
                "temperature": 0.0,  # Use temperature 0 for deterministic output
                "max_tokens": 10,
            }

            # Generate with first LoRA adapter
            input_data_1 = input_data_base.copy()
            input_data_1["lora_adapter_name"] = "test_lora_1"
            result_1 = lora_backend.generate(input_data_1)

            # Generate with second LoRA adapter
            input_data_2 = input_data_base.copy()
            input_data_2["lora_adapter_name"] = "test_lora_2"
            result_2 = lora_backend.generate(input_data_2)

            # Generate with base model (no LoRA)
            result_base = lora_backend.generate(input_data_base)

            # Verify all generations are successful
            assert "error" not in result_1
            assert "error" not in result_2
            assert "error" not in result_base

            # Extract generated content
            content_1 = result_1["choices"][0]["message"]["content"]
            content_2 = result_2["choices"][0]["message"]["content"]
            content_base = result_base["choices"][0]["message"]["content"]

            # Verify they're all valid strings
            assert isinstance(content_1, str)
            assert isinstance(content_2, str)
            assert isinstance(content_base, str)

            # Verify generate was called 3 times
            assert mock_generate.call_count == 3


def test_lora_adapter_persistence_across_generations(
    lora_backend, mock_peft_model
):
    """Test that LoRA adapter remains loaded across multiple generations."""
    with patch(
        "sllm.backends.transformers_backend.load_model",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.load_lora",
        return_value=mock_peft_model,
    ), patch(
        "sllm.backends.transformers_backend.TransformersBackend._tokenize"
    ) as mock_tokenize:
        lora_backend.init_backend()

        # Setup tokenizer mock
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        mock_inputs = tokenizer("test prompt", return_tensors="pt")
        mock_tokenize.return_value = mock_inputs

        # Simulate LoRA adapter loaded
        mock_peft_model.peft_config["test_lora"] = {"adapter_name": "test_lora"}

        # Mock model generation
        with patch.object(mock_peft_model, "generate") as mock_generate:
            # Create mock output tokens
            mock_output = torch.tensor([[1, 2, 3, 4, 5, 6]])
            mock_generate.return_value = mock_output

            input_data = {
                "model": "facebook/opt-125m",
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ],
                "temperature": 0.7,
                "max_tokens": 5,
                "lora_adapter_name": "test_lora",
            }

            # Generate multiple times
            for i in range(3):
                result = lora_backend.generate(input_data)
                assert "error" not in result
                assert "choices" in result and len(result["choices"]) == 1

                # Verify adapter is still loaded
                assert hasattr(lora_backend.model, "peft_config")
                assert "test_lora" in lora_backend.model.peft_config

            # Verify generate was called 3 times
            assert mock_generate.call_count == 3
