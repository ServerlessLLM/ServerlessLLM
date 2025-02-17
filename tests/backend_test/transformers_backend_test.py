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

import peft
import pytest
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from sllm.serve.backends.backend_utils import BackendStatus
from sllm.serve.backends.transformers_backend import (
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
        "sllm.serve.backends.transformers_backend.load_model"
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
        mock_load_model.assert_called_once_with(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
            hf_model_class=hf_model_class,
        )


def test_init_encoder_backend(encoder_backend, encoder_config):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model"
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
        mock_load_model.assert_called_once_with(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            storage_path=storage_path,
            hf_model_class=hf_model_class,
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
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
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
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
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
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
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
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
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
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
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
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=encoder,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._encoder_tokenize",
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
        "sllm.serve.backends.transformers_backend.load_model",
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


def test_fine_tuning(transformers_backend, model, tokenizer):
    with patch(
        "sllm.serve.backends.transformers_backend.load_model",
        return_value=model,
    ), patch(
        "sllm.serve.backends.transformers_backend.TransformersBackend._tokenize",
        return_value=tokenizer,
    ):
        transformers_backend.init_backend()
        input = {
            "model": "facebook/opt-125m",
            "ft_backend": "peft",
            "dataset_config": {
                "dataset_source": "hf_hub",
                "hf_dataset_name": "fka/awesome-chatgpt-prompts",
                "tokenization_field": "prompt",
                "split": "train[:10%]",
                "data_files": "",
                "extension_type": "",
            },
            "lora_config": {
                "r": 4,
                "lora_alpha": 1,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.05,
                "bias": "lora_only",
                "task_type": "CAUSAL_LM",
            },
            "training_config": {
                "auto_find_batch_size": True,
                "num_train_epochs": 2,
                "learning_rate": 0.0001,
                "use_cpu": False,
            },
        }
        result = transformers_backend.fine_tuning(input)
        assert "error" not in result
        assert "model" in result and "lora_save_path" in result
